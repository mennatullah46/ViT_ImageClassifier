##🦊 Vision Transformer (ViT) for Animal Image Classification

#A PyTorch implementation of a Vision Transformer (ViT) model trained to classify 18 animal species using a custom dataset.

This project includes:

Custom ViT architecture (patch embedding, transformer encoder, MSA, MLP blocks)

Pretrained weight loading from google/vit-base-patch16-224-in21k

Full training & evaluation pipeline

Data augmentation, normalization, and DataLoader setup

Cosine Annealing LR scheduler + AdamW optimizer

Model saving & loading utilities

📁 Project Structure

├── AnimalTrainData/
│   ├── train.csv
│   ├── AnimalTrainData/
│       ├── <image files>
├── vit_model_final2.pth
├── train_vit.py
└── README.md

##📦 Requirements

#Install dependencies:

pip install torch torchvision torchaudio
pip install transformers
pip install torchinfo
pip install pandas numpy scikit-learn pillow

##🐾 Dataset

#The dataset is defined by a CSV file:

ImageID, Class
img_001.jpg, beaver
img_002.jpg, dolphin
...

Each image path is automatically mapped to:

AnimalTrainData/AnimalTrainData/<ImageID>

Class Mapping (18 classes)

beaver, butterfly, cougar, crab, crayfish, crocodile,
dolphin, dragonfly, elephant, flamingo, kangaroo,
leopard, llama, lobster, octopus, pigeon, rhino, scorpion

##🔧 Data Preprocessing & Augmentation

Training Transformations

RandomResizedCrop

Horizontal Flip

Rotation (15°)

Color Jitter

ImageNet normalization

Validation Transformations

Resize to 224×224

ImageNet normalization

##🧱 Model Architecture

#This project implements a custom Vision Transformer from scratch.

#Components:

PatchEmbedding using Conv2D

CLS token + positional embeddings

Multi-Head Self Attention (MSA)

MLP block with GELU

12 Transformer Encoder layers

Linear classifier head

Pretrained Initialization

Loads pretrained weights from:

google/vit-base-patch16-224-in21k

Classifier and pooler layers are excluded.

##🚀 Training

#The training loop includes:

CrossEntropyLoss

AdamW optimizer (LR = 5e‑5)

CosineAnnealingLR scheduler

Accuracy + loss tracking

Run training:

optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs=200)
train_model(model, train_loader, optimizer, scheduler, 200, device)

##🧪 Validation

#Evaluation uses:

CrossEntropyLoss

Accuracy over validation set

test_model(model, val_loader, device)

##💾 Saving & Loading the Model

#Save:

torch.save(model.state_dict(), "vit_model_final2.pth")

#Load:

model.load_state_dict(torch.load("vit_model_final2.pth"))

##📤 Export to Google Drive (Colab)

shutil.move("/content/vit_model_final2.pth",
            "/content/drive/MyDrive/vit_model_final2.pth")

##📊 Model Summary

summary(model, input_size=(1, 3, 224, 224))

##📝 Notes

Batch size defaults to 64 (adjust based on GPU memory).

Training for 200 epochs recommended.

Model is initialized with pretrained ViT weights but trained end‑to‑end.
