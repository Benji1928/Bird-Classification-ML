import os
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image


class BirdDataset(td.Dataset):  # td.Dataset is a class from torch.utils.data
    def __init__(self, root_dir, mode="train", image_size=(224,224), transform=None):
        super().__init__()
        self.image_size = image_size
        self.mode = mode
        self.transform = transform
        
        # Loading Dataset from Train Folder     
        self.images_dir = os.path.join(root_dir, "Train")  # safer to use root_dir here
        
        # Loading .txt file (train.txt / val.txt)
        txt_path = os.path.join(root_dir, f'{mode}.txt')
        print("Text File:", txt_path)
        self.data = pd.read_csv(txt_path, sep=" ", header=None, names=["file_path", "class"])

        # Store transform
        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(self.image_size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        
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
        return "BirdDataset: (mode='{}', image_size={})".format(self.mode, self.image_size)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = os.path.join(self.images_dir, self.data.iloc[idx]['file_path'])
        img = Image.open(img_path).convert('RGB') 
        label = self.data.iloc[idx]['class']

        # Apply transform if provided, else default
        transform_to_apply = self.transform or (self.train_transform if "train" in self.mode else self.val_transform)
        img = transform_to_apply(img)


        return img, label

    def number_of_classes(self):
        return self.data['class'].max() + 1 