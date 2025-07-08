import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import logging

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer model for fingerprint feature extraction.
    Combines EfficientNet for local features and Swin Transformer for global features.
    """
    def __init__(self, num_classes=500, embedding_dim=512):
        super(HybridCNNTransformer, self).__init__()
        # CNN Backbone (EfficientNet)
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn_features = nn.Sequential(*list(self.cnn.children())[:-2])
        self.spatial_attention = SpatialAttention(in_channels=1280)

        # Transformer Backbone (Swin)
        self.vit = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

        # Fusion and Embedding Layers
        self.fc = nn.Sequential(
            nn.Linear(1280 + 768, embedding_dim), # EfficientNet-B0 features + Swin-Tiny features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    # def forward(self, x):
    #     # CNN Path
    #     cnn_features = self.cnn_features(x)
    #     cnn_features = self.spatial_attention(cnn_features)
    #     cnn_pooled = F.adaptive_avg_pool2d(cnn_features, (1, 1)).flatten(1)

    #     # Transformer Path
    #     vit_features = self.vit.forward_features(x)
    #     vit_pooled = vit_features.mean(dim=1) # Global average pooling

    #     # Fusion
    #     fused_features = torch.cat([cnn_pooled, vit_pooled], dim=1)
    #     embedding = self.fc(fused_features)
    #     classification = self.classifier(embedding)

    #     return {'embedding': embedding, 'classification': classification}

    def forward(self, x):
      # CNN Path
      cnn_features = self.cnn_features(x)
      cnn_features = self.spatial_attention(cnn_features)

      cnn_pooled = F.adaptive_avg_pool2d(cnn_features, (1, 1)).flatten(1)

      vit_features = self.vit.forward_features(x)

      if vit_features.dim() == 4:

          vit_pooled = vit_features.mean(dim=[1, 2])
      else:

          vit_pooled = vit_features.mean(dim=1)

      # Fusion
      fused_features = torch.cat([cnn_pooled, vit_pooled], dim=1)
      embedding = self.fc(fused_features)
      classification = self.classifier(embedding)

      return {'embedding': embedding, 'classification': classification}
    

class FingerprintDataset(Dataset):
    """
    A generic dataset for loading fingerprint images.
    It expects preprocessed images and applies only standard transforms.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Return image and its corresponding numerical label
        return image, torch.tensor(self.labels[idx], dtype=torch.long)

def setup_training_dataset(data_dir, transform):
    """
    Prepares the training dataset from the non-latent (rolled) prints directory.
    """
    print(f"Loading NON-LATENT prints for TRAINING from: {data_dir}")
    image_paths = glob.glob(os.path.join(data_dir, '*.bmp'))
    if not image_paths:
        raise FileNotFoundError(f"No training images (.bmp) found in {data_dir}.")

    # Create numerical labels based on the unique filenames
    finger_ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in image_paths])
    id_to_label = {finger_id: i for i, finger_id in enumerate(finger_ids)}
    labels = [id_to_label[os.path.splitext(os.path.basename(p))[0]] for p in image_paths]

    print(f"Loaded {len(image_paths)} images for training across {len(id_to_label)} classes.")
    return FingerprintDataset(image_paths, labels, transform=transform)

def train_classification_model(model, train_loader, num_epochs, device, model_save_path):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0

    print("Starting model training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss, train_preds, train_labels_list = 0, [], []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['classification'], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs['classification'], dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_labels_list, train_preds)
        print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        if train_acc > best_acc:
            best_acc = train_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f'---> New best model saved with accuracy {best_acc:.4f}')
    print("Training complete.")


if __name__ == "__main__":
    NON_LATENT_DIR = '/content/drive/MyDrive/latent_dataset/IIITD_Latent_Mated_1000ppi_Enhanced'
    LATENT_DIR = '/content/drive/MyDrive/latent_dataset/IITD_latent_ROI/full_processed_context_aware'
    MODEL_SAVE_PATH = '/content/drive/MyDrive/latent_model/fingerprint_model_final.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Define transforms for training (with augmentation)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = setup_training_dataset(NON_LATENT_DIR, transform=train_transform)
    except FileNotFoundError as e:
        logging.error(f"Error loading training data: {e}"); exit()

    num_classes = len(np.unique(train_dataset.labels))
    model = HybridCNNTransformer(num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    train_classification_model(model, train_loader, num_epochs=100, device=DEVICE, model_save_path=MODEL_SAVE_PATH)