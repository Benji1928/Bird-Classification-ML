import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import os
import pandas as pd
import gradio as gr
import numpy as np
import nntools
from data import BirdDataset


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def criterion(self, y, d):
        return self.cross_entropy(y, d)

class VGG16(Classifier):
    def __init__(self, num_classes, fine_tuning=False, dropout_p=0.5):
        super().__init__()

        vgg = tv.models.vgg16_bn(pretrained=True)
        
        # Apply freezing 
        for name, param in vgg.features.named_parameters():
            layer_idx = int(name.split('.')[0])
            # Freezing layers 0, 3, 7 (conv1_1, conv2_1, conv3_1)
            if layer_idx in [0, 3, 7]:
                param.requires_grad = False
        
        self.features = vgg.features

        num_ftrs = 25088 
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f = self.features(x).view(x.shape[0], -1) 
        y = self.classifier(f)
        return y


class Resnet18Transfer(Classifier):
    def __init__(self, num_classes, fine_tuning=False, dropout_p=0.5):
        super().__init__()
        resnet = tv.models.resnet18(pretrained=True)
        
        # Freeze or unfreeze layers based on fine_tuning flag
        for param in resnet.parameters():
            param.requires_grad = fine_tuning
        
        num_ftrs = resnet.fc.in_features
        
        # Replace the final fully connected layer with dropout + new classifier
        resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes)
        )
        
        self.classifier = resnet

    def forward(self, x):
        return self.classifier(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 200 
MODELS = {}



def load_model(name, model_class, model_dir):
    model = model_class(NUM_CLASSES).to(device)
    model_path = os.path.join(model_dir, "checkpoint.pth.tar")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine the key for the state_dict based on nntools.Experiment saving format
        if isinstance(checkpoint, dict) and 'Net' in checkpoint:
             model.load_state_dict(checkpoint['Net'])
        else:
             # Fallback for directly saved state_dict
             model.load_state_dict(checkpoint)
             
        model.eval()
        print(f"{name} model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: {name} checkpoint not found at {model_path}. Skipping.")
        return None
    except RuntimeError as e:
        print(f"Error loading {name} state dict: {e}. Skipping.")
        
        
        
# Determine the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
        return None



# Load VGG16 (birdclass1)
VGG_MODEL_DIR = "birdclass1" 
MODELS['VGG16'] = load_model('VGG16', VGG16, VGG_MODEL_DIR)

# Load ResNet18 (birdclass2)
RESNET_MODEL_DIR = "birdclass2" 
MODELS['ResNet18'] = load_model('ResNet18', Resnet18Transfer, RESNET_MODEL_DIR)


transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),  # Resize to 224x224
    tv.transforms.CenterCrop((224, 224)), # Center crop
    tv.transforms.ToTensor(),           # Convert to tensor
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) # Normalize
])



def get_class_names(num_classes):
    # This should be replaced with your actual class name list
    return [f"Bird_Class_{i+1}" for i in range(num_classes)]

CLASS_NAMES = get_class_names(NUM_CLASSES)




def predict_image(arch_name: str, img: Image.Image) -> dict:
    """
    Predicts the class of the input image using the selected model.
    """
    if img is None:
        return {name: 0.0 for name in CLASS_NAMES}
    
    model = MODELS.get(arch_name)
    if model is None:
        return {"Error": f"{arch_name} model is not loaded."}

    # Apply preprocessing
    img_tensor = transform(img).unsqueeze(0)

    # Perform inference
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)

    # Post-process: apply softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Convert to dictionary
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
    
    return confidences



# Define the model choices for the dropdown
model_choices = list(MODELS.keys())
# Filter out any models that failed to load
model_choices = [name for name in model_choices if MODELS[name] is not None]



if not model_choices:
    raise RuntimeError("No models were loaded successfully. Check MODEL_PATH and file contents.")

    

model_dropdown = gr.Dropdown(
    label="Select Model Architecture",
    choices=model_choices,
    value=model_choices[0], # Default to the first loaded model
    interactive=True
)

image_input = gr.Image(
    type="pil", 
    label="Upload Bird Image (224x224 will be used)", 
    width=224, 
    height=224
)

label_output = gr.Label(num_top_classes=5)



# Create the Gradio Interface
demo = gr.Interface(
    fn=predict_image,
    inputs=[model_dropdown, image_input],
    outputs=label_output,
    title="Bird Species Classification: VGG16 vs. ResNet18",
    description="Select a pre-trained model and upload an image of a bird to get the top 5 predicted species.",
    examples=[] # Empty list for simplicity, but you can add valid paths here
)



if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict_image,
        inputs=[model_dropdown, image_input],
        outputs=label_output,
        title="Bird Species Classification: VGG16 vs. ResNet18",
        description="Select a pre-trained model and upload an image of a bird to get the top 5 predicted species.",
        examples=[]
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)
