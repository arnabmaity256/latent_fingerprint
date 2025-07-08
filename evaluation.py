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
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

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
    
def get_normalized_id_from_filename(filename):
    """
    Extracts the subject and finger ID from a filename, normalizing finger names
    like 'LS' to 'LL' and 'RS' to 'RL'.
    e.g., '10_LS_Card_1.xyt' -> '10_LL'
    e.g., '10_RL.bmp' -> '10_RL'
    """
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    parts = name_without_ext.split('_')

    if len(parts) >= 2:
        subject_id = parts[0]
        finger_id = parts[1]

        # Normalize finger names
        finger_mapping = {'LS': 'LL', 'RS': 'RL'}
        normalized_finger_id = finger_mapping.get(finger_id, finger_id)

        return f"{subject_id}_{normalized_finger_id}"
    else:
        return parts[0] if parts else None
    
def setup_evaluation_datasets(latent_dir, non_latent_dir, transform):
    """
    Prepares the probe (latent) and gallery (non-latent) datasets for evaluation.
    This version uses the first two parts of the filename as the ID for evaluation.
    """
    print(f"Loading LATENT prints for EVALUATION from: {latent_dir}")
    latent_paths = glob.glob(os.path.join(latent_dir, '*.jpg'))
    if not latent_paths:
        raise FileNotFoundError(f"No latent images (.jpg) found for evaluation in {latent_dir}.")

    latent_ids = [get_normalized_id_from_filename(p) for p in latent_paths]

    # Create a mapping from ID to numerical label
    all_ids = sorted(list(set(latent_ids)))
    id_to_label = {id_val: i for i, id_val in enumerate(all_ids)}

    latent_labels = [id_to_label[id_val] for id_val in latent_ids]
    probe_dataset = FingerprintDataset(latent_paths, latent_labels, transform)
    print(f"Found {len(probe_dataset)} latent probe prints.")

    print(f"Loading NON-LATENT prints for EVALUATION GALLERY from: {non_latent_dir}")
    non_latent_paths = glob.glob(os.path.join(non_latent_dir, '*.bmp'))
    if not non_latent_paths:
        raise FileNotFoundError(f"No non-latent images (.bmp) found for gallery in {non_latent_dir}.")


    non_latent_ids = [get_normalized_id_from_filename(p) for p in non_latent_paths]


    all_ids = sorted(list(set(all_ids + non_latent_ids)))
    id_to_label = {id_val: i for i, id_val in enumerate(all_ids)}

    non_latent_labels = [id_to_label[id_val] for id_val in non_latent_ids]
    gallery_dataset = FingerprintDataset(non_latent_paths, non_latent_labels, transform)
    print(f"Found {len(gallery_dataset)} non-latent gallery prints.")

    return probe_dataset, gallery_dataset


def calculate_genuine_imposter_scores(probe_embeddings, probe_labels, gallery_embeddings, gallery_labels):
    genuine_scores = []
    imposter_scores = []
    print("Calculating genuine and impostor scores...")
    for i in tqdm(range(len(probe_embeddings)), desc="Calculating Scores"):
        probe_embedding = probe_embeddings[i]
        probe_label = probe_labels[i]
        similarities = cosine_similarity([probe_embedding], gallery_embeddings)[0]
        for j, score in enumerate(similarities):
            if probe_label == gallery_labels[j]:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)
    return np.array(genuine_scores), np.array(imposter_scores)

def calculate_eer(genuine_scores, impostor_scores):
    """
    Calculates the Equal Error Rate (EER) for a set of genuine and impostor scores.
    """
    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)

    eer_threshold_index = np.argmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eer_threshold_index]
    eer = (fpr[eer_threshold_index] + (1 - tpr[eer_threshold_index])) / 2.0
    return eer, eer_threshold

def evaluate_latent_identification(model, probe_loader, gallery_loader, device, top_k=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    model.to(device)
    model.eval()
    print("Extracting gallery embeddings...")
    gallery_embeddings, gallery_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(gallery_loader, desc="Gallery"):
            images = images.to(device)
            outputs_dict = model(images)
            embedding = outputs_dict['embedding']
            gallery_embeddings.append(embedding.cpu().numpy())
            gallery_labels.append(labels.numpy())
    gallery_embeddings = np.vstack(gallery_embeddings)
    gallery_labels = np.concatenate(gallery_labels)
    print(f"Enrolled {len(gallery_embeddings)} fingerprints in the gallery.")
    print("Matching probes against gallery...")
    probe_embeddings, probe_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(probe_loader, desc="Probes"):
            images = images.to(device)
            outputs_dict = model(images)
            embedding = outputs_dict['embedding']
            probe_embeddings.append(embedding.cpu().numpy())
            probe_labels.append(labels.numpy())
    probe_embeddings = np.vstack(probe_embeddings)
    probe_labels = np.concatenate(probe_labels)
    genuine_scores, impostor_scores = calculate_genuine_imposter_scores(
        probe_embeddings, probe_labels, gallery_embeddings, gallery_labels
    )


    try:
        pd.DataFrame(genuine_scores, columns=['score']).to_csv('genuine_scores.csv', index=False)
        print("Genuine scores saved to 'genuine_scores_cnn.csv'")
        pd.DataFrame(impostor_scores, columns=['score']).to_csv('impostor_scores.csv', index=False)
        print("Impostor scores saved to 'impostor_scores_cnn.csv'")
    except Exception as e:
        print(f"Error saving scores to CSV: {e}")


    print("\n" + "="*40)
    print("--- Score Distribution ---")
    print(f"Total Genuine Scores: {len(genuine_scores)}")
    print(f"Total Impostor Scores: {len(impostor_scores)}")
    print("\n" + "="*40 + "\n")

    # Plot histogram of scores
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine Scores')
    plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor Scores')
    plt.title('Distribution of Genuine and Impostor Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('score_distribution_histogram.png')
    plt.close()
    print("Histogram saved as score_distribution_histogram.png")

    # Plot ROC curve
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()
    print("ROC curve saved as roc_curve.png")


    eer, eer_threshold = calculate_eer(genuine_scores, impostor_scores)
    print("\n" + "="*40)
    print("--- EER Calculation ---")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"EER Threshold: {eer_threshold:.4f}")
    print("="*40 + "\n")


    correct_ranks = []
    similarity_matrix = cosine_similarity(probe_embeddings, gallery_embeddings)
    for i in range(len(probe_embeddings)):
        true_label = probe_labels[i]
        scores = similarity_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]
        sorted_gallery_labels = gallery_labels[sorted_indices]
        try:
            rank = list(sorted_gallery_labels).index(true_label) + 1
            correct_ranks.append(rank)
        except ValueError:
            correct_ranks.append(float('inf'))
    correct_ranks = np.array(correct_ranks)
    print("\n" + "="*40)
    print("--- Latent Identification Performance ---")
    print(f"Total Latent Probes Tested: {len(correct_ranks)}")
    for k in top_k:
        accuracy = np.mean(correct_ranks <= k) * 100
        print(f"Rank-{k} Accuracy: {accuracy:.2f}%\n")
    print("="*40 + "\n")


if __name__ == "__main__":
    NON_LATENT_DIR = '/content/drive/MyDrive/latent_dataset/IIITD_Latent_Mated_1000ppi_Enhanced'
    LATENT_DIR = '/content/drive/MyDrive/latent_dataset/IITD_latent_ROI/full_processed_context_aware'
    MODEL_SAVE_PATH = '/content/drive/MyDrive/latent_model/fingerprint_model_final.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\nProceeding to evaluation...")
    if not os.path.exists(MODEL_SAVE_PATH):
        logging.error(f"Model file not found at '{MODEL_SAVE_PATH}'. Please train first.")
    else:

        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Setup evaluation datasets
        probe_dataset, gallery_dataset = setup_evaluation_datasets(
            latent_dir=LATENT_DIR,
            non_latent_dir=NON_LATENT_DIR,
            transform=eval_transform
        )

        probe_loader = DataLoader(probe_dataset, batch_size=16, shuffle=False, num_workers=2)
        gallery_loader = DataLoader(gallery_dataset, batch_size=16, shuffle=False, num_workers=2)

        # Load the trained model
        trained_model = HybridCNNTransformer(num_classes=150)
        trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(DEVICE)))

    evaluate_latent_identification(trained_model, probe_loader, gallery_loader, device=DEVICE)