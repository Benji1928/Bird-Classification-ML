import os
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image

class BirdDataset(td.Dataset): 
    def __init__(self, root_dir, mode="train", image_size=(224,224), transform=None):
        super().__init__()
        self.image_size = image_size
        self.mode = mode
        self.custom_transform = transform  # Rename to avoid confusion
        
        # Determine which txt file to use based on mode
        if mode == "train":
            txt_path = os.path.join(root_dir, 'train_split.txt')
        elif mode == "val":
            txt_path = os.path.join(root_dir, 'val_split.txt')
        else:  # mode == "test"
            txt_path = os.path.join(root_dir, 'test.txt')
        
        # Set images directory - train and val both use Train folder
        if mode in ["train", "val"]:
            self.images_dir = os.path.join(root_dir, 'Train')
        else:  # mode == "test"
            self.images_dir = os.path.join(root_dir, 'Test')
        
        # Loading .txt file
        print(f"Text File: {txt_path}")
        print(f"Images Directory: {self.images_dir}")
        self.data = pd.read_csv(txt_path, sep=" ", header=None, names=["file_path", "class"])
        print(f"Loaded {len(self.data)} samples for {mode} mode")
        
        # Define training transform (with augmentation)
        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        ])
        
        # Define validation/test transform (no augmentation)
        self.val_transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.CenterCrop(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"BirdDataset(mode='{self.mode}', size={len(self)}, image_size={self.image_size})"
    
    def __getitem__(self, idx):
        # Get image path and label
        img_path = os.path.join(self.images_dir, self.data.iloc[idx]['file_path'])
        
        # Try to load image with error handling
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found: {img_path}")
            # Create a black placeholder image
            img = Image.new('RGB', self.image_size, color='black')
        
        label = self.data.iloc[idx]['class']
        
        # Choose which transform to apply
        if self.custom_transform is not None:
            # Use custom transform if provided
            img = self.custom_transform(img)
        elif self.mode == "train":
            # Use training transform (with augmentation)
            img = self.train_transform(img)
        else:  # mode == "val" or "test"
            # Use validation transform (no augmentation)
            img = self.val_transform(img)
        
        return img, label
    
    def number_of_classes(self):
        return self.data['class'].max() + 1