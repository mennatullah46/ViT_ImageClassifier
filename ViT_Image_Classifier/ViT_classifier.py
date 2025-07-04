pip install torchinfo

import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchinfo import summary
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from math import cos, pi
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import ViTModel, ViTConfig
from torch.nn.functional import cross_entropy

# ------------------------------
# 1) Load the dataset
# ------------------------------
data = pd.read_csv('AnimalTrainData/train.csv')
data['image_path'] = data['ImageID'].apply(lambda x: f'AnimalTrainData/AnimalTrainData/{x}')

# Train-validation split before augmentation
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Class'])

# Class Mapping (for encoding labels)
class_mapping = {
    'beaver': 0, 'butterfly': 1, 'cougar': 2, 'crab': 3, 'crayfish': 4, 'crocodile': 5,
    'dolphin': 6, 'dragonfly': 7, 'elephant': 8, 'flamingo': 9, 'kangaroo': 10, 'leopard': 11,
    'llama': 12, 'lobster': 13, 'octopus': 14, 'pigeon': 15, 'rhino': 16, 'scorpion': 17
}

# Encode labels
y_train = train_df['Class'].values
y_valid = val_df['Class'].values

y_train_encoded = np.array([class_mapping[label] for label in y_train])
y_valid_encoded = np.array([class_mapping[label] for label in y_valid])

train_df['Class'] = y_train_encoded
val_df['Class'] = y_valid_encoded

# ------------------------------
# 2) Define transformations
#    (Added Normalization, toned-down rotation)
# ------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    # Added normalization with ImageNet stats
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Same normalization for validation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ------------------------------
# 3) Create Dataset & DataLoaders
# ------------------------------
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['Class']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.int64)
        return image, label

train_dataset = ImageDataset(train_df, transform=train_transform)
val_dataset   = ImageDataset(val_df,   transform=valid_transform)

# Increased batch size to 64 if your GPU can handle it; else keep 32
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

# Check DataLoader sizes
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Example: Iterate through DataLoader (just 1 batch)
for images, labels in train_loader:
    print("Batch of images shape:", images.shape)
    print("Batch of labels shape:", labels.shape)
    break

# ------------------------------
# 4) Vision Transformer Classes
# ------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)  # [batch_size, num_patches, embedding_dim]

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, dropout_rate=0.5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x)
        x = self.dropout(attn_output)
        return x

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout_rate=0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_size, embedding_dim)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.mlp(x_norm)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, dropout_rate=0.2):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim, num_heads, dropout_rate)
        self.mlp_block = MLPBlock(embedding_dim, mlp_size, dropout_rate)

    def forward(self, x):
        # Residual connection around MSA
        x = self.msa_block(x) + x
        # Residual connection around MLP
        x = self.mlp_block(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, num_classes=18, embedding_dim=768, num_heads=12,
                 num_layers=12, image_size=224, patch_size=16):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, embedding_dim=embedding_dim)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embedding_dim)) # Class token
        self.pos_embed   = nn.Parameter(torch.randn(1, (image_size // patch_size)**2 + 1, embedding_dim))
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, num_heads) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        # Take the CLS token for classification
        x = x[:, 0]
        x = self.classifier(x)
        return x

# ------------------------------
# 5) Pretrained Weight Loading
# ------------------------------
def initialize_scratch_vit_with_pretrained(model_name, num_classes, device):
    # Load pre-trained config + weights
    pretrained_config = ViTConfig.from_pretrained(model_name)
    pretrained_vit = ViTModel.from_pretrained(model_name, config=pretrained_config)

    # Create your custom ViT
    scratch_vit = ViT(
        num_classes=num_classes,
        embedding_dim=pretrained_config.hidden_size,
        num_heads=pretrained_config.num_attention_heads,
        num_layers=pretrained_config.num_hidden_layers,
        image_size=224,
        patch_size=16
    )

    # Merge state dict
    scratch_dict = scratch_vit.state_dict()
    pretrained_dict = pretrained_vit.state_dict()

    # Filter out unnecessary keys
    filtered_pretrained_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in scratch_dict and "classifier" not in k and "pooler" not in k
    }
    scratch_dict.update(filtered_pretrained_dict)
    scratch_vit.load_state_dict(scratch_dict)

    return scratch_vit.to(device)

# Instantiate model
model_name = "google/vit-base-patch16-224-in21k"
num_classes = 18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialize_scratch_vit_with_pretrained(model_name, num_classes, device)
summary(model, input_size=(1, 3, 224, 224))

# ------------------------------
# 6) Optimizer & Scheduler
#    (Lower LR to 5e-5, typical for fine-tuning)
# ------------------------------
def get_optimizer_and_scheduler(model, total_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    return optimizer, scheduler

# ------------------------------
# 7) Train & Test Functions
# ------------------------------
def train_model(model, train_loader, optimizer, scheduler, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_corrects += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_corrects / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        scheduler.step()

def test_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    val_loss = val_loss / len(val_loader)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    return val_loss, accuracy

# ------------------------------
# 8) Train & Evaluate
# ------------------------------
optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs=200)
train_model(model, train_loader, optimizer, scheduler, 200, device)
test_model(model, val_loader, device)

# 2. Save the model after training
torch.save(model.state_dict(), "vit_model_final2.pth")
print("Final model saved as vit_model_final.pth")

# 4. Reload and test
model.load_state_dict(torch.load("vit_model_final2.pth"))
print("Loaded model from vit_model_final.pth")
test_model(model, val_loader, device)

import shutil

# Source file in Colab
source = "/content/vit_model_final2.pth"  # Replace with your file name

# Destination in Google Drive
destination = "/content/drive/MyDrive/vit_model_final2.pth"  # Replace with your desired path in Drive

# Move the file
shutil.move(source, destination)
print(f"{source} has been moved to {destination}")

