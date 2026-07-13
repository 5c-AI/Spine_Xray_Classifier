"""
Fracture Detection Training Script using EfficientDet with Command Line Arguments

Installation:
pip install effdet

Usage Examples:
# Basic training with default parameters
python fracture_efficientdet_training_args.py --data_dir Data --csv_path csvs/frac_ann.csv --model_name tf_efficientdet_d7x --normal_limit 1000

#Avaailable arguments:
--data_dir: Path to the data directory
--csv_path: Path to the CSV file containing the annotations
--model_name: Name of the model to use
--normal_limit: Maximum number of normal images to use
--batch_size: Batch size for training
--max_epochs: Maximum number of epochs to train for
--run_evaluation: Whether to run evaluation on the test set
--train_split: Ratio of data for training (default: 0.9)
--test_split: Ratio of data for testing (default: 0.05)
--val_split: Ratio of data for validation (default: 0.05)
--eval_conf_threshold: Confidence threshold for evaluation
--eval_iou_threshold: IoU threshold for evaluation
--eval_seed: Random seed for reproducible splits (default: 42)

# Training with custom data paths
python fracture_efficientdet_training_args.py --data_dir /path/to/data --csv_path /path/to/annotations.csv


Available EfficientDet Models:
- tf_efficientdet_d0, tf_efficientdet_d1, tf_efficientdet_d2, tf_efficientdet_d3
- tf_efficientdet_d4, tf_efficientdet_d5, tf_efficientdet_d6, tf_efficientdet_d7, tf_efficientdet_d7x

Tips:
1. Start with smaller models (d0-d3) for faster experimentation
2. Use larger batch sizes with smaller models if GPU memory allows
3. Monitor training with TensorBoard logs for optimal hyperparameter tuning
4. The evaluation script automatically finds and tests your latest models
5. Use consistent test sets (same seed) to compare different models fairly
6. Check the TP/FP/TN/FN directories to understand model behavior
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import csv

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json
import ast
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import argparse

# EfficientDet related imports
import effdet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
import timm

# For inference and evaluation
from ensemble_boxes import weighted_boxes_fusion
import random
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import glob
from tqdm import tqdm

# ================================================================================================
# HELPERS
# ================================================================================================

def _path_column(df: pd.DataFrame) -> str:
    """Return CSV column name for image path: 'path', 'image_path', or 'study_path'."""
    for col in ('path', 'image_path', 'study_path'):
        if col in df.columns:
            return col
    raise ValueError("CSV must contain one of: path, image_path, study_path")

# ================================================================================================
# DATASET CLASSES
# ================================================================================================

class PathologyDatasetAdaptor:
    """Dataset adaptor for pathology detection. CSV: path/image_path/study_path, label (Abnormal/Normal), bbox."""
    
    def __init__(self, abnormal_dir_path: str, normal_dir_path: str, csv_path: str):
        self.abnormal_dir_path = abnormal_dir_path
        self.normal_dir_path = normal_dir_path
        self.csv_path = csv_path
        
        # Initialize corrupted image tracking
        self.corrupted_count = 0
        self.corrupted_log_file = f"corrupted_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Write initial header to log file
        with open(self.corrupted_log_file, 'w') as f:
            f.write(f"Corrupted Images Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
        
        # Load CSV data
        self.annotations_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.annotations_df)} annotations from CSV")
        
        path_col = _path_column(self.annotations_df)
        if 'label' in self.annotations_df.columns:
            labels = self.annotations_df['label'].astype(str).str.strip().str.lower()
            abnormal_df = self.annotations_df[labels == 'abnormal']
            normal_df = self.annotations_df[labels == 'normal']
            abnormal_images = abnormal_df[path_col].unique().tolist()
            csv_normal_images = normal_df[path_col].unique().tolist() if len(normal_df) else []
        else:
            abnormal_images = self.annotations_df[path_col].unique().tolist()
            csv_normal_images = []
        
        existing_abnormal_images = []
        for img_path in abnormal_images:
            candidate_paths = []
            if os.path.isabs(img_path):
                candidate_paths.append(img_path)
            candidate_paths.append(os.path.join(self.abnormal_dir_path, img_path.lstrip('/')))
            candidate_paths.append(os.path.join(self.abnormal_dir_path, os.path.basename(img_path)))
            candidate_paths.append(os.path.join(self.abnormal_dir_path, img_path.replace('/', '_')))
            full_image_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
            if full_image_path is not None:
                existing_abnormal_images.append(('abnormal', img_path))
            else:
                print(f"Warning: Abnormal image not found: {img_path}")
        
        existing_normal_images = []
        if csv_normal_images:
            for img_path in csv_normal_images:
                candidate_paths = []
                if os.path.isabs(img_path):
                    candidate_paths.append(img_path)
                candidate_paths.append(os.path.join(self.normal_dir_path, img_path.lstrip('/')))
                candidate_paths.append(os.path.join(self.normal_dir_path, os.path.basename(img_path)))
                candidate_paths.append(os.path.join(self.normal_dir_path, img_path.replace('/', '_')))
                full_image_path = next((p for p in candidate_paths if os.path.exists(p)), None)
                if full_image_path is not None:
                    existing_normal_images.append(('normal', img_path))
                else:
                    print(f"Warning: Normal image from CSV not found: {img_path}")
        
        normal_images_from_dir = []
        # Skip filesystem walk when single dir and we already have normals from CSV (faster)
        need_walk = os.path.exists(self.normal_dir_path) and (
            self.normal_dir_path != self.abnormal_dir_path or not existing_normal_images
        )
        if need_walk:
            abnormal_basenames = {os.path.basename(p) for _, p in existing_abnormal_images}
            csv_normal_set = set(csv_normal_images) if csv_normal_images else set()
            for root, _, files in os.walk(self.normal_dir_path):
                for file in files:
                    if not file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        continue
                    relative_path = os.path.relpath(os.path.join(root, file), self.normal_dir_path)
                    if self.normal_dir_path == self.abnormal_dir_path:
                        if os.path.basename(relative_path) in abnormal_basenames:
                            continue
                    if relative_path in csv_normal_set:
                        continue
                    normal_images_from_dir.append(('normal', relative_path))
        
        self.images = existing_abnormal_images + existing_normal_images + normal_images_from_dir
        print(f"Found {len(existing_abnormal_images)} abnormal images out of {len(abnormal_images)} total")
        print(f"Found {len(existing_normal_images)} normal images from CSV, {len(normal_images_from_dir)} from directory")
        print(f"Total dataset size: {len(self.images)} images")
        print(f"Corrupted images will be logged to: {self.corrupted_log_file}")
        
    def log_corrupted_image(self, image_path: str, error_msg: str):
        """Log corrupted image information to file"""
        self.corrupted_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.corrupted_log_file, 'a') as f:
            f.write(f"[{timestamp}] Corrupted Image #{self.corrupted_count}\n")
            f.write(f"Path: {image_path}\n")
            f.write(f"Error: {error_msg}\n")
            f.write("-" * 40 + "\n")
        
        print(f"Skipped corrupted image #{self.corrupted_count}: {image_path}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def get_image_and_labels_by_idx(self, index: int) -> Tuple[Image.Image, np.ndarray, np.ndarray, str]:
        image_type, image_path = self.images[index]
        if image_type == 'abnormal':
            candidate_paths = []
            if os.path.isabs(image_path):
                candidate_paths.append(image_path)
            candidate_paths.append(os.path.join(self.abnormal_dir_path, image_path.lstrip('/')))
            candidate_paths.append(os.path.join(self.abnormal_dir_path, os.path.basename(image_path)))
            candidate_paths.append(os.path.join(self.abnormal_dir_path, image_path.replace('/', '_')))
            full_image_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
        else:
            full_image_path = os.path.join(self.normal_dir_path, image_path)
        
        # Load image with error handling
        try:
            image = Image.open(full_image_path).convert("RGB")
            width, height = image.size
        except (OSError, IOError, Exception) as e:
            # Log the corrupted image
            self.log_corrupted_image(full_image_path, str(e))
            
            # Return a dummy black image to continue training
            image = Image.new('RGB', (512, 512), color='black')
            width, height = 512, 512
            
            # Return empty bboxes for corrupted images
            pascal_bboxes = np.array([], dtype=np.float32).reshape(0, 4)
            class_labels = np.array([], dtype=np.float32)
            return image, pascal_bboxes, class_labels, str(index)
        
        bboxes = []
        if image_type == 'abnormal':
            path_col = _path_column(self.annotations_df)
            image_annotations = self.annotations_df[self.annotations_df[path_col] == image_path]
            for _, row in image_annotations.iterrows():
                bbox_str = row['bbox']
                if bbox_str and bbox_str != '':
                    try:
                        bbox_list = ast.literal_eval(bbox_str)
                        for bbox_dict in bbox_list:
                            x = (bbox_dict['x'] / 100) * width
                            y = (bbox_dict['y'] / 100) * height
                            w = (bbox_dict['width'] / 100) * width
                            h = (bbox_dict['height'] / 100) * height
                            xmin = max(0, min(x, width - 1))
                            ymin = max(0, min(y, height - 1))
                            xmax = max(xmin + 1, min(x + w, width))
                            ymax = max(ymin + 1, min(y + h, height))
                            bboxes.append([xmin, ymin, xmax, ymax])
                    except Exception as e:
                        print(f"Error parsing bbox for image {image_path}: {e}")
                        continue
        if len(bboxes) > 0:
            pascal_bboxes = np.array(bboxes, dtype=np.float32)
            class_labels = np.ones(len(bboxes), dtype=np.float32)
        else:
            pascal_bboxes = np.array([], dtype=np.float32).reshape(0, 4)
            class_labels = np.array([], dtype=np.float32)
        return image, pascal_bboxes, class_labels, str(index)
    
    def show_image(self, index: int):
        """Visualize image with bounding boxes"""
        image, bboxes, labels, image_id = self.get_image_and_labels_by_idx(index)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(f'Image ID: {image_id} - {len(bboxes)} boxes')
        plt.show()

# ================================================================================================
# MODEL HELPER FUNCTIONS
# ================================================================================================

def create_efficientdet_model(num_classes: int, 
                            model_name: str = 'tf_efficientdet_d5',
                            pretrained: bool = True) -> DetBenchTrain:
    """Create EfficientDet model with image size from config"""
    
    # Get configuration for the specified model
    config = get_efficientdet_config(model_name)
    
    # Get image size from config
    image_size = config.image_size[0]
    print(f"Using image size from {model_name} config: {image_size}")
    
    # Handle DenseNet backbone compatibility - remove drop_path_rate if using DenseNet
    # DenseNet models don't support drop_path_rate parameter, so we need to remove it
    # to avoid "TypeError: DenseNet.__init__() got an unexpected keyword argument 'drop_path_rate'"
    if 'densenet' in model_name and 'backbone_args' in config and 'drop_path_rate' in config.backbone_args:
        print(f"Removing drop_path_rate for DenseNet backbone compatibility")
        config.backbone_args = {k: v for k, v in config.backbone_args.items() if k != 'drop_path_rate'}
    
    # Ensure image_size is a tuple (height, width)
    if isinstance(image_size, int):
        image_size_tuple = (image_size, image_size)
    else:
        image_size_tuple = image_size
    
    config.update({
        'num_classes': num_classes,
        'image_size': image_size_tuple
    })
    
    # Create model
    net = EfficientDet(config, pretrained_backbone=pretrained)
    
    # Replace classification head for custom number of classes
    net.class_net = HeadNet(config, num_outputs=num_classes)
    
    return DetBenchTrain(net), image_size

# ================================================================================================
# PYTORCH DATASET AND TRANSFORMS
# ================================================================================================

class EfficientDetDataset(Dataset):
    """Dataset class for EfficientDet training"""
    
    def __init__(self, dataset_adaptor, transforms=None):
        self.ds = dataset_adaptor
        self.transforms = transforms
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        image, pascal_bboxes, class_labels, image_id = self.ds.get_image_and_labels_by_idx(index)
        
        # Convert PIL image to numpy array
        image = np.array(image)
        
        # Handle empty annotations
        if len(pascal_bboxes) == 0:
            # Create dummy annotation to avoid training issues
            h, w = image.shape[:2]
            pascal_bboxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            class_labels = np.array([0], dtype=np.float32)  # background class
        
        # Convert pascal VOC to YXYX format (required by EfficientDet)
        yxyx_bboxes = pascal_bboxes.copy()
        yxyx_bboxes[:, [0, 1, 2, 3]] = yxyx_bboxes[:, [1, 0, 3, 2]]  # xyxy -> yxyx
        
        if self.transforms:
            # Apply albumentations transforms
            sample = {
                'image': image,
                'bboxes': pascal_bboxes,
                'labels': class_labels
            }
            
            try:
                sample = self.transforms(**sample)
                image = sample['image']
                if len(sample['bboxes']) > 0:
                    yxyx_bboxes = np.array(sample['bboxes'])
                    yxyx_bboxes[:, [0, 1, 2, 3]] = yxyx_bboxes[:, [1, 0, 3, 2]]  # xyxy -> yxyx
                    class_labels = np.array(sample['labels'])
                else:
                    # Handle case where transforms removed all boxes
                    yxyx_bboxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
                    class_labels = np.array([0], dtype=np.float32)
            except Exception as e:
                print(f"Transform error for image {image_id}: {e}")
                # Fall back to original image
                image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        
        # Convert to tensors
        # Handle image dimensions properly
        if isinstance(image, torch.Tensor):
            img_height, img_width = image.shape[1], image.shape[2]
        else:
            img_height, img_width = image.shape[0], image.shape[1]
            
        targets = {
            'bbox': torch.tensor(yxyx_bboxes, dtype=torch.float32),
            'cls': torch.tensor(class_labels, dtype=torch.long),
            'img_size': torch.tensor([img_width, img_height], dtype=torch.float32),
            'img_scale': torch.tensor(1.0, dtype=torch.float32)
        }
        
        return image, targets, image_id

def get_train_transforms(image_size: int = 832):
    """Training transforms with medical image normalization"""
    # Handle case where image_size might be a list [height, width]
    if isinstance(image_size, (list, tuple)):
        height, width = image_size[0], image_size[1]
    else:
        height = width = image_size
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.Resize(height=height, width=width, p=1.0),
        # Use custom normalization for medical images
        A.Normalize([0.20793004, 0.20824375, 0.20862524], 
                   [0.23613826, 0.23636451, 0.23669834]),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))

def get_valid_transforms(image_size: int = 832):
    """Validation transforms"""
    # Handle case where image_size might be a list [height, width]
    if isinstance(image_size, (list, tuple)):
        height, width = image_size[0], image_size[1]
    else:
        height = width = image_size
    
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        A.Normalize([0.20793004, 0.20824375, 0.20862524], 
                   [0.23613826, 0.23636451, 0.23669834]),

                   
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

class EfficientDetDataModule(pl.LightningDataModule):
    """Lightning DataModule for EfficientDet"""
    
    def __init__(self, 
                 train_dataset_adaptor,
                 validation_dataset_adaptor,
                 train_transforms=None,
                 valid_transforms=None,
                 num_workers: int = 8,
                 batch_size: int = 1):
        super().__init__()
        
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def train_dataloader(self) -> DataLoader:
        train_dataset = EfficientDetDataset(
            dataset_adaptor=self.train_ds,
            transforms=self.train_transforms
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        return train_loader
    
    def val_dataloader(self) -> DataLoader:
        valid_dataset = EfficientDetDataset(
            dataset_adaptor=self.valid_ds,
            transforms=self.valid_transforms
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        return valid_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()
        
        boxes = []
        labels = []
        img_sizes = []
        img_scales = []
        
        for i, target in enumerate(targets):
            boxes.append(target['bbox'])
            labels.append(target['cls'])
            img_sizes.append(target['img_size'])
            img_scales.append(target['img_scale'])
        
        # Stack img_size and img_scale tensors properly
        img_size_tensor = torch.stack(img_sizes)
        img_scale_tensor = torch.stack(img_scales)
        
        return images, {'bbox': boxes, 'cls': labels, 'img_size': img_size_tensor, 'img_scale': img_scale_tensor}, image_ids

# ================================================================================================
# PYTORCH LIGHTNING MODULE
# ================================================================================================

class EfficientDetModel(pl.LightningModule):
    """Lightning Module for EfficientDet"""
    
    def __init__(self, 
                 num_classes: int = 1,
                 learning_rate: float = 0.0001,
                 wbf_iou_threshold: float = 0.44,
                 model_name: str = 'tf_efficientdet_d5'):
        super().__init__()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.model_name = model_name
        
        self.model, self.image_size = create_efficientdet_model(
            num_classes=num_classes,
            model_name=model_name
        )
        
        print(f"Model {model_name} initialized with image size: {self.image_size}")
        
        self.save_hyperparameters()
    
    @property
    def image_dim(self):
        """Get single image dimension from image_size (handles both int and list)"""
        if isinstance(self.image_size, (list, tuple)):
            return self.image_size[0]  # Assume square images, take first dimension
        return self.image_size
    
    def forward(self, images, targets):
        return self.model(images, targets)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        
        losses = self.model(images, targets)
        
        self.log("train_loss", losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_class_loss", losses['class_loss'], on_step=True, on_epoch=True)
        self.log("train_box_loss", losses['box_loss'], on_step=True, on_epoch=True)
        
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        
        losses = self.model(images, targets)
        
        self.log("val_loss", losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_class_loss", losses['class_loss'], on_step=True, on_epoch=True)
        self.log("val_box_loss", losses['box_loss'], on_step=True, on_epoch=True)
        
        return losses['loss']

# ================================================================================================
# MAIN TRAINING FUNCTION
# ================================================================================================

def train_detection(
    data_dir: str,
    csv_path: str,
    abnormal_dir_name: str = None,
    model_name: str = 'tf_efficientdet_d5',
    batch_size: int = 4,
    max_epochs: int = 50,
    learning_rate: float = 0.0001,
    train_split: float = 0.9,
    test_split: float = 0.05,
    val_split: float = 0.05,
    num_workers: int = 8,
    run_evaluation: bool = True,
    normal_limit: int = None,
    eval_conf_threshold: float = 0.3,
    eval_iou_threshold: float = 0.5,
    eval_seed: int = 42
):
    """
    Train EfficientDet for any pathology. All images can be in data_dir (abnormal_dir_name=None),
    or use a subfolder with abnormal_dir_name (e.g. "Bronchiectasis") and optional normal/ subfolder.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"runs/{model_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Single directory: both abnormal and normal in data_dir (e.g. /home/ai-user/disk/xray_data/chest/)
    if not abnormal_dir_name or abnormal_dir_name.strip() == "":
        abnormal_dir = os.path.abspath(data_dir)
        normal_dir = abnormal_dir
        print("Using single data directory for both abnormal and normal images.")
    else:
        possible_normal_names = ["normal", "neg", "negative"]
        abnormal_dir = os.path.join(data_dir, abnormal_dir_name)
        if not os.path.exists(abnormal_dir):
            if os.path.exists(data_dir) and abnormal_dir_name in data_dir:
                abnormal_dir = data_dir
            else:
                raise ValueError(f"Abnormal directory not found: {abnormal_dir}. Use no --abnormal_dir if all images are in data_dir.")
        normal_dir = None
        for name in possible_normal_names:
            candidate = os.path.join(data_dir, name)
            if os.path.exists(candidate):
                normal_dir = candidate
                break
        if normal_dir is None and os.path.dirname(data_dir):
            for name in possible_normal_names:
                candidate = os.path.join(os.path.dirname(data_dir), name)
                if os.path.exists(candidate):
                    normal_dir = candidate
                    break
        if normal_dir is None:
            normal_dir = abnormal_dir  # fallback: same dir
    
    print(f"Data directory: {data_dir}")
    print(f"Abnormal directory: {abnormal_dir}")
    print(f"Normal directory: {normal_dir}")
    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found: {csv_path}")
    
    # Get image size from model config
    config = get_efficientdet_config(model_name)
    image_size = config.image_size[0]
    print(f"Using image size {image_size} for model {model_name}")
    
    # Extract single dimension for transforms (handle both int and list)
    if isinstance(image_size, (list, tuple)):
        image_dim = image_size[0]  # Assume square images, take first dimension
    else:
        image_dim = image_size
    
    # ================================================================================================
    # STEP 1: CREATE TEST SET BEFORE TRAINING (PREVENT DATA LEAKAGE)
    # ================================================================================================
    
    print("\n" + "="*80)
    print("STEP 1: CREATING TEST SET (BEFORE TRAINING)")
    print("="*80)
    
    # Set random seed for reproducible splits
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)
    
    # Verify split ratios sum to 1.0
    total_split = train_split + test_split + val_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, but got: train={train_split}, test={test_split}, val={val_split} (sum={total_split})")
    
    annotations_df = pd.read_csv(csv_path)
    path_col = _path_column(annotations_df)
    if 'label' in annotations_df.columns:
        labels = annotations_df['label'].astype(str).str.strip().str.lower()
        abnormal_df = annotations_df[labels == 'abnormal']
        normal_df = annotations_df[labels == 'normal']
        abnormal_image_list = abnormal_df[path_col].unique().tolist()
        csv_normal_image_list = normal_df[path_col].unique().tolist() if len(normal_df) else []
    else:
        abnormal_image_list = annotations_df[path_col].unique().tolist()
        csv_normal_image_list = []
    
    existing_abnormal_images = []
    for img_path in abnormal_image_list:
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        candidates.append(os.path.join(abnormal_dir, img_path.lstrip('/')))
        candidates.append(os.path.join(abnormal_dir, os.path.basename(img_path)))
        candidates.append(os.path.join(abnormal_dir, img_path.replace('/', '_')))
        if next((p for p in candidates if os.path.exists(p)), None) is not None:
            existing_abnormal_images.append(img_path)
    
    existing_normal_images_from_csv = []
    for img_path in csv_normal_image_list:
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        candidates.append(os.path.join(normal_dir or '', img_path.lstrip('/')))
        candidates.append(os.path.join(normal_dir or '', os.path.basename(img_path)))
        candidates.append(os.path.join(normal_dir or '', img_path.replace('/', '_')))
        if next((p for p in candidates if os.path.exists(p)), None) is not None:
            existing_normal_images_from_csv.append(img_path)
    
    all_normal_images = list(existing_normal_images_from_csv)
    # Skip filesystem walk when single dir and we already have normals from CSV (faster)
    need_walk = normal_dir and os.path.exists(normal_dir) and (
        normal_dir != abnormal_dir or not existing_normal_images_from_csv
    )
    if need_walk:
        abnormal_basenames = {os.path.basename(p) for p in existing_abnormal_images}
        all_normal_set = set(all_normal_images)
        csv_normal_set = set(csv_normal_image_list) if csv_normal_image_list else set()
        for root, _, files in os.walk(normal_dir):
            for file in files:
                if not file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    continue
                relative_path = os.path.relpath(os.path.join(root, file), normal_dir)
                if relative_path in all_normal_set:
                    continue
                if os.path.basename(relative_path) in abnormal_basenames:
                    continue
                if relative_path in csv_normal_set:
                    continue
                all_normal_set.add(relative_path)
                all_normal_images.append(relative_path)
    
    print(f"Found {len(existing_abnormal_images)} abnormal images, {len(all_normal_images)} normal images (before limits)")
    random.shuffle(existing_abnormal_images)
    random.shuffle(all_normal_images)
    
    test_abnormal_size = int(len(existing_abnormal_images) * test_split)
    test_normal_size = int(len(all_normal_images) * test_split)
    remaining_abnormal = len(existing_abnormal_images) - test_abnormal_size
    remaining_normal = len(all_normal_images) - test_normal_size
    val_abnormal_size = int(remaining_abnormal * (val_split / (1 - test_split)))
    val_normal_size = int(remaining_normal * (val_split / (1 - test_split)))
    
    test_abnormal = existing_abnormal_images[:test_abnormal_size]
    test_normal = all_normal_images[:test_normal_size]
    remaining_abnormal_images = existing_abnormal_images[test_abnormal_size:]
    remaining_normal_images = all_normal_images[test_normal_size:]
    val_abnormal = remaining_abnormal_images[:val_abnormal_size]
    val_normal = remaining_normal_images[:val_normal_size]
    train_abnormal = remaining_abnormal_images[val_abnormal_size:]
    train_normal_all = remaining_normal_images[val_normal_size:]
    
    if normal_limit is not None and len(train_normal_all) > normal_limit:
        random.shuffle(train_normal_all)
        train_normal = train_normal_all[:normal_limit]
        print(f"Applied normal_limit={normal_limit}; training normals: {len(train_normal)}")
    else:
        train_normal = train_normal_all
    
    print(f"\nDataset Split (seed={eval_seed}):")
    print(f"  Train: {len(train_abnormal)} abnormal + {len(train_normal)} normal")
    print(f"  Val:   {len(val_abnormal)} abnormal + {len(val_normal)} normal")
    print(f"  Test:  {len(test_abnormal)} abnormal + {len(test_normal)} normal")
    
    test_data = {
        'abnormal': test_abnormal,
        'normal': test_normal,
        'abnormal_dir': abnormal_dir,
        'normal_dir': normal_dir or '',
        'annotations_df': annotations_df,
        'test_split': test_split
    }
    
    test_set_file = os.path.join(output_dir, 'test_set.json')
    with open(test_set_file, 'w') as f:
        json.dump({
            'test_abnormal': test_abnormal,
            'test_normal': test_normal,
            'train_split': train_split,
            'test_split': test_split,
            'val_split': val_split,
            'normal_limit': normal_limit,
            'eval_seed': eval_seed,
            'timestamp': timestamp,
            'abnormal_dir_name': abnormal_dir_name
        }, f, indent=2)
    print(f"Test set saved to: {test_set_file}")
    
    # ================================================================================================
    # STEP 2: CREATE TRAINING DATASET (EXCLUDING TEST SET)
    # ================================================================================================
    
    print("\n" + "="*80)
    print("STEP 2: CREATING TRAINING DATASET (EXCLUDING TEST SET)")
    print("="*80)
    
    # Optimize for A100 Tensor Cores
    torch.set_float32_matmul_precision('medium')  # Better performance on A100
    
    train_images = [('abnormal', p) for p in train_abnormal] + [('normal', p) for p in train_normal]
    val_images = [('abnormal', p) for p in val_abnormal] + [('normal', p) for p in val_normal]
    
    corrupted_log_file = f"corrupted_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(corrupted_log_file, 'w') as f:
        f.write(f"Corrupted Images Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
    
    def _make_adaptor(images_list):
        a = PathologyDatasetAdaptor.__new__(PathologyDatasetAdaptor)
        a.abnormal_dir_path = abnormal_dir
        a.normal_dir_path = normal_dir or ''
        a.csv_path = csv_path
        a.annotations_df = annotations_df
        a.images = images_list
        a.corrupted_count = 0
        a.corrupted_log_file = corrupted_log_file
        return a
    
    train_adaptor = _make_adaptor(train_images)
    val_adaptor = _make_adaptor(val_images)
    
    print(f"Training set: {len(train_images)} images, Validation set: {len(val_images)} images")
    
    # ================================================================================================
    # STEP 4: TRAINING (USING ONLY TRAIN+VAL, TEST SET COMPLETELY ISOLATED)
    # ================================================================================================
    
    print("\n" + "="*80)
    print("STEP 4: STARTING TRAINING (TEST SET ISOLATED)")
    print("="*80)
    
    # Create data module
    dm = EfficientDetDataModule(
        train_dataset_adaptor=train_adaptor,
        validation_dataset_adaptor=val_adaptor,
        train_transforms=get_train_transforms(image_dim),
        valid_transforms=get_valid_transforms(image_dim),
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    model = EfficientDetModel(
        num_classes=1,
        learning_rate=learning_rate,
        model_name=model_name
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f'{model_name}_detection-{{epoch:02d}}-{{val_loss:.3f}}',
        monitor='val_loss',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        mode='min',
        verbose=True
    )
    
    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs"
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=10.0,
        accumulate_grad_batches=3,  # Effective batch size = batch_size * 3
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_model_summary=True
    )
    
    # Train model
    trainer.fit(model, dm)
    
    with open(corrupted_log_file, 'a') as f:
        f.write(f"\n" + "=" * 60 + "\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Corrupted skipped - train: {train_adaptor.corrupted_count}, val: {val_adaptor.corrupted_count}\n")
    print(f"Corrupted images skipped: train={train_adaptor.corrupted_count}, val={val_adaptor.corrupted_count}")
    
    final_model_path = os.path.join(output_dir, f'{model_name}_detection_final.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    print(f"Training completed! Best model saved in: {output_dir}")
    print(f"Final model saved as: {final_model_path}")
    
    # ================================================================================================
    # STEP 5: EVALUATION ON HELD-OUT TEST SET
    # ================================================================================================
    
    # Run automatic evaluation if requested
    evaluation_results = None
    if run_evaluation:
        try:
            print("\n" + "="*80)
            print("STEP 5: EVALUATING ON HELD-OUT TEST SET")
            print("="*80)
            
            evaluation_results = evaluate_trained_model_with_test_data(
                model=model,
                trainer=trainer,
                output_dir=output_dir,
                test_data=test_data,
                conf_threshold=eval_conf_threshold,
                batch_size=4,
                iou_threshold=eval_iou_threshold
            )
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            print("Training completed successfully, but evaluation encountered errors.")
    
    return model, trainer, evaluation_results

# ================================================================================================
# COMMAND LINE ARGUMENT PARSER
# ================================================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pathology detection training (EfficientDet) - change --abnormal_dir for your pathology')
    parser.add_argument('--data_dir', type=str, default='/root/Data_utils/DATA_SPINE',
                        help='Path to data directory')
    parser.add_argument('--csv_path', type=str, default='/root/SPINE_PATHOLOGIES/OSTEO/csv/osteo_new.csv',
                        help='Path to CSV (columns: path/image_path/study_path, label, bbox)')
    parser.add_argument('--abnormal_dir', type=str, default=None,
                        help='Subfolder for abnormal images under data_dir. If not set, all images are in data_dir (single folder)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='tf_efficientdet_d5',
                      help='EfficientDet model name (default: tf_efficientdet_d5)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training (default: 8)')
    parser.add_argument('--max_epochs', type=int, default=100,
                      help='Maximum number of epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate (default: 0.0001)')
    parser.add_argument('--train_split', type=float, default=0.9,
                      help='Training split ratio (default: 0.9)')
    parser.add_argument('--test_split', type=float, default=0.05,
                      help='Test split ratio (default: 0.05)')
    parser.add_argument('--val_split', type=float, default=0.05,
                      help='Validation split ratio (default: 0.05)')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of data loading workers (default: auto-detect)')
    
    # Evaluation arguments
    parser.add_argument('--run_evaluation', action='store_true', default=True,
                      help='Run automatic evaluation after training (default: True)')
    parser.add_argument('--skip_evaluation', action='store_true', default=False,
                      help='Skip automatic evaluation after training')
    parser.add_argument('--normal_limit', type=int, default=None,
                      help='Maximum number of normal images to consider during evaluation (default: no limit)')
    parser.add_argument('--eval_conf_threshold', type=float, default=0.3,
                      help='Confidence threshold for evaluation (default: 0.3)')
    parser.add_argument('--eval_iou_threshold', type=float, default=0.5,
                      help='IoU threshold for evaluation (default: 0.5)')
    parser.add_argument('--eval_seed', type=int, default=42,
                      help='Random seed for reproducible evaluation splits (default: 42)')
    
    return parser.parse_args()

# ================================================================================================
# EVALUATION HELPER FUNCTIONS
# ================================================================================================

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x1, y1, x2, y2] format"""
    # Ensure boxes are in correct format
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there's intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def match_predictions_to_ground_truth(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes using IoU threshold
    Returns matches with IoU values and unmatched predictions/ground truths
    """
    matches = []
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    
    # Sort predictions by confidence (highest first)
    if len(pred_boxes) > 0:
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        for pred_idx in sorted_indices:
            if pred_matched[pred_idx]:
                continue
                
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                    
                iou = calculate_iou(pred_boxes[pred_idx], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If IoU is above threshold, create match
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matches.append({
                    'pred_idx': pred_idx,
                    'gt_idx': best_gt_idx,
                    'iou': best_iou,
                    'confidence': pred_scores[pred_idx],
                    'pred_box': pred_boxes[pred_idx],
                    'gt_box': gt_boxes[best_gt_idx]
                })
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
    
    # Get unmatched predictions and ground truths
    unmatched_preds = [i for i, matched in enumerate(pred_matched) if not matched]
    unmatched_gts = [i for i, matched in enumerate(gt_matched) if not matched]
    
    return matches, unmatched_preds, unmatched_gts

def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation"""
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Calculate AP using trapezoidal rule
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def calculate_coco_metrics(all_matches, all_unmatched_preds, all_unmatched_gts, 
                          total_gt_boxes, iou_thresholds=None):
    """Calculate COCO-style metrics including AP@0.5, AP@0.5:0.95"""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5, 0.55, 0.6, ..., 0.95
    
    metrics = {}
    aps = []
    
    for iou_thresh in iou_thresholds:
        # Filter matches by IoU threshold
        valid_matches = [m for m in all_matches if m['iou'] >= iou_thresh]
        
        if len(valid_matches) == 0 and total_gt_boxes == 0:
            ap = 1.0  # Perfect if no GT and no predictions
        elif len(valid_matches) == 0:
            ap = 0.0  # No matches
        else:
            # Sort by confidence
            valid_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate precision and recall at each detection
            tp = np.zeros(len(valid_matches))
            fp = np.zeros(len(valid_matches))
            
            for i, match in enumerate(valid_matches):
                tp[i] = 1  # All valid matches are TP by definition
            
            # Add FP from unmatched predictions with high confidence
            all_confs = [m['confidence'] for m in valid_matches]
            
            # Calculate cumulative TP and FP
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # Calculate precision and recall
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            recalls = tp_cumsum / (total_gt_boxes + 1e-8)
            
            # Calculate AP
            ap = calculate_ap(precisions, recalls)
        
        aps.append(ap)
        metrics[f'AP@{iou_thresh:.2f}'] = ap
    
    # Calculate mean AP over all IoU thresholds
    metrics['AP@0.5:0.95'] = np.mean(aps)
    metrics['AP@0.5'] = aps[0] if len(aps) > 0 else 0.0
    
    return metrics

def run_inference_on_test_data(model: EfficientDetModel, test_data: dict, 
                              output_dir: str, conf_threshold: float = 0.3, batch_size: int = 4,
                              iou_threshold: float = 0.5):
    """Run comprehensive IoU-based inference evaluation on test dataset"""
    print("Running comprehensive IoU-based evaluation on test dataset...")
    
    evaluation_results = {}
    device = next(model.parameters()).device
    model.eval()
    
    # Collect all matches for COCO metrics
    all_matches = []
    all_unmatched_preds = []
    all_unmatched_gts = []
    total_gt_boxes = 0
    
    # Collect all image paths with their types
    all_images = []
    
    for img_path in test_data['abnormal']:
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        candidates.append(os.path.join(test_data['abnormal_dir'], img_path.lstrip('/')))
        candidates.append(os.path.join(test_data['abnormal_dir'], os.path.basename(img_path)))
        candidates.append(os.path.join(test_data['abnormal_dir'], img_path.replace('/', '_')))
        final_path = next((p for p in candidates if os.path.exists(p)), None)
        if final_path is not None:
            img_id = f"abnormal_{hash(img_path) % 100000}"
            all_images.append({
                'img_path': img_path,
                'full_image_path': final_path,
                'img_id': img_id,
                'is_abnormal': True
            })
    
    for img_path in test_data['normal']:
        full_image_path = os.path.join(test_data['normal_dir'], img_path)
        if os.path.exists(full_image_path):
            img_id = f"normal_{hash(img_path) % 100000}"
            all_images.append({
                'img_path': img_path,
                'full_image_path': full_image_path,
                'img_id': img_id,
                'is_abnormal': False
            })
    
    print(f"Running IoU-based evaluation on {len(all_images)} images with batch size {batch_size}")
    print(f"Confidence threshold: {conf_threshold}, IoU threshold: {iou_threshold}")
    
    # Process images in batches with progress bar
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), batch_size), desc="Processing batches"):
            batch_images = all_images[i:i + batch_size]
            
            # Prepare batch
            batch_tensors = []
            batch_info = []
            
            for img_info in batch_images:
                try:
                    # Load and preprocess image
                    image = Image.open(img_info['full_image_path']).convert("RGB")
                    original_size = image.size  # (width, height)
                    
                    # Use validation transforms
                    transforms = get_valid_transforms(model.image_dim)
                    image_array = np.array(image)
                    
                    # Apply transforms
                    transformed = transforms(image=image_array, bboxes=[], labels=[])
                    image_tensor = transformed['image']
                    
                    # Ensure it's a tensor
                    if isinstance(image_tensor, np.ndarray):
                        image_tensor = torch.from_numpy(image_tensor).float()
                    
                    batch_tensors.append(image_tensor)
                    batch_info.append({
                        'img_info': img_info,
                        'original_size': original_size
                    })
                    
                except Exception as e:
                    print(f"Error loading image {img_info['full_image_path']}: {e}")
                    # Add dummy evaluation result for failed images
                    evaluation_results[img_info['img_id']] = {
                        'predictions': {'boxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])},
                        'ground_truth_boxes': np.array([]).reshape(0, 4),
                        'matches': [],
                        'unmatched_preds': [],
                        'unmatched_gts': [],
                        'mean_iou': 0.0,
                        'max_confidence': 0.0,
                        'num_predictions': 0,
                        'num_gt_boxes': 0
                    }
                    continue
            
            if not batch_tensors:
                continue
                
            # Stack batch tensors
            batch_tensor = torch.stack(batch_tensors).to(device)
            
            # Create dummy targets for inference
            batch_size_actual = batch_tensor.shape[0]
            dummy_targets = {
                'bbox': [torch.zeros((1, 4), dtype=torch.float32, device=device)] * batch_size_actual,
                'cls': [torch.zeros(1, dtype=torch.long, device=device)] * batch_size_actual,
                'img_size': torch.stack([torch.tensor([model.image_dim, model.image_dim], 
                                                    dtype=torch.float32, device=device)] * batch_size_actual),
                'img_scale': torch.stack([torch.tensor(1.0, dtype=torch.float32, device=device)] * batch_size_actual)
            }
            
            # Run forward pass
            try:
                outputs = model(batch_tensor, dummy_targets)
                
                # Process outputs for each image in batch
                for idx, info in enumerate(batch_info):
                    img_info = info['img_info']
                    original_size = info['original_size']
                    
                    # Extract predictions
                    predictions = extract_single_prediction(
                        outputs, idx, original_size, model.image_dim
                    )
                    
                    if img_info['is_abnormal']:
                        gt_boxes = get_ground_truth_boxes(
                            img_info['img_path'], 
                            test_data['annotations_df'], 
                            original_size
                        )
                    else:
                        gt_boxes = np.array([]).reshape(0, 4)  # Normal images have no GT boxes
                    
                    # Perform IoU-based matching
                    pred_boxes = predictions.get('boxes', [])
                    pred_scores = predictions.get('scores', [])
                    
                    if len(pred_boxes) > 0 and len(pred_scores) > 0:
                        # Convert to numpy arrays and ensure they're 1D
                        pred_scores_array = np.array(pred_scores)
                        if pred_scores_array.ndim > 1:
                            pred_scores_array = pred_scores_array.flatten()
                        
                        # Filter predictions by confidence
                        high_conf_mask = pred_scores_array >= conf_threshold
                        high_conf_indices = np.where(high_conf_mask)[0]
                        
                        filtered_boxes = [pred_boxes[i] for i in high_conf_indices]
                        filtered_scores = [pred_scores_array[i] for i in high_conf_indices]
                    else:
                        filtered_boxes = []
                        filtered_scores = []
                    
                    # Match predictions to ground truth
                    if len(filtered_boxes) > 0 and len(gt_boxes) > 0:
                        matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
                            np.array(filtered_boxes), np.array(filtered_scores), gt_boxes, iou_threshold
                        )
                    else:
                        matches = []
                        unmatched_preds = list(range(len(filtered_boxes)))
                        unmatched_gts = list(range(len(gt_boxes)))
                    
                    # Calculate metrics for this image
                    ious = [m['iou'] for m in matches]
                    mean_iou = np.mean(ious) if ious else 0.0
                    
                    # Safe max calculation for pred_scores
                    if len(pred_scores) > 0:
                        # Convert to numpy array and handle multi-dimensional arrays
                        pred_scores_array = np.array(pred_scores)
                        if pred_scores_array.ndim > 1:
                            pred_scores_array = pred_scores_array.flatten()
                        max_confidence = float(np.max(pred_scores_array))
                    else:
                        max_confidence = 0.0
                    
                    # Store evaluation results
                    img_result = {
                        'predictions': predictions,
                        'ground_truth_boxes': gt_boxes,
                        'matches': matches,
                        'unmatched_preds': unmatched_preds,
                        'unmatched_gts': unmatched_gts,
                        'mean_iou': mean_iou,
                        'max_confidence': max_confidence,
                        'num_predictions': len(pred_boxes),
                        'num_gt_boxes': len(gt_boxes)
                    }
                    
                    evaluation_results[img_info['img_id']] = img_result
                    
                    # Accumulate for COCO metrics
                    all_matches.extend(matches)
                    all_unmatched_preds.extend(unmatched_preds)
                    all_unmatched_gts.extend(unmatched_gts)
                    total_gt_boxes += len(gt_boxes)
                    
                    create_enhanced_visualization(
                        img_info['full_image_path'], img_result, img_info['img_id'],
                        img_info['is_abnormal'], gt_boxes, output_dir=output_dir,
                        conf_threshold=conf_threshold, iou_threshold=iou_threshold
                    )
                    
            except Exception as e:
                print(f"Error during batch inference: {e}")
                # Add empty evaluation results for this batch
                for info in batch_info:
                    evaluation_results[info['img_info']['img_id']] = {
                        'predictions': {'boxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])},
                        'ground_truth_boxes': np.array([]).reshape(0, 4),
                        'matches': [],
                        'unmatched_preds': [],
                        'unmatched_gts': [],
                        'mean_iou': 0.0,
                        'max_confidence': 0.0,
                        'num_predictions': 0,
                        'num_gt_boxes': 0
                    }
    
    # Calculate COCO metrics
    print("\nCalculating COCO metrics...")
    coco_metrics = calculate_coco_metrics(all_matches, all_unmatched_preds, all_unmatched_gts, total_gt_boxes)
    
    print(f"IoU-based evaluation completed on {len(evaluation_results)} images")
    print(f"Total ground truth boxes: {total_gt_boxes}")
    print(f"Total matches found: {len(all_matches)}")
    print(f"COCO Metrics:")
    for metric_name, value in coco_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    return evaluation_results, coco_metrics

def extract_single_prediction(outputs, idx: int, original_size: tuple, model_img_dim: int):
    """Extract prediction for a single image from batch outputs"""
    try:
        # Process outputs
        if isinstance(outputs, dict) and 'detections' in outputs:
            detections = outputs['detections']
            
            if hasattr(detections[idx], 'cpu'):
                detection = detections[idx].cpu().numpy()
            else:
                detection = detections[idx]
            
            # Detection format: [x1, y1, x2, y2, confidence, class]
            if len(detection) > 0 and detection.shape[1] >= 6:
                boxes = detection[:, :4]
                scores = detection[:, 4]
                labels = detection[:, 5]
                
                # Scale boxes back to original image size
                orig_w, orig_h = original_size
                scale_x = orig_w / model_img_dim
                scale_y = orig_h / model_img_dim
                
                boxes[:, [0, 2]] *= scale_x  # x coordinates
                boxes[:, [1, 3]] *= scale_y  # y coordinates
                
                return {
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }
            else:
                return {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([])
                }
        else:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }
            
    except Exception as e:
        print(f"Error extracting prediction for image {idx}: {e}")
        return {
            'boxes': np.array([]),
            'scores': np.array([]),
            'labels': np.array([])
        }

def evaluate_trained_model_with_test_data(model: EfficientDetModel, trainer: Trainer, output_dir: str,
                          test_data: dict, conf_threshold: float = 0.3, batch_size: int = 4,
                          iou_threshold: float = 0.5):
    """
    Comprehensive evaluation with IoU-based metrics and COCO evaluation
    
    This function uses the test set that was created BEFORE training to ensure
    no data leakage and provide unbiased evaluation results with proper object detection metrics.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE IoU-BASED EVALUATION (NO DATA LEAKAGE)")
    print("="*80)
    print(f"Test set size: {len(test_data['abnormal']) + len(test_data['normal'])} images")
    print(f"  - Abnormal: {len(test_data['abnormal'])}, Normal: {len(test_data['normal'])}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    # Run comprehensive IoU-based evaluation
    evaluation_results, coco_metrics = run_inference_on_test_data(
        model, test_data, output_dir, conf_threshold, batch_size=batch_size, iou_threshold=iou_threshold
    )
    
    # Calculate IoU-based metrics
    print("\n" + "="*60)
    print("CALCULATING IoU-BASED DETECTION METRICS")
    print("="*60)
    
    # Collect statistics for IoU-based evaluation
    total_gt_boxes = 0
    total_predictions = 0
    tp_detections = 0  # True positive detections (IoU >= threshold)
    fp_detections = 0  # False positive detections (no matching GT or IoU < threshold)
    fn_detections = 0  # False negative detections (GT boxes not matched)
    
    # Image-level classification statistics
    tp_images = 0  # Fracture images correctly detected (with good IoU)
    fp_images = 0  # Normal images incorrectly classified as fracture
    tn_images = 0  # Normal images correctly classified
    fn_images = 0  # Fracture images missed or detected with poor IoU
    
    # IoU statistics
    all_ious = []
    matched_confidences = []
    
    for img_id, img_result in evaluation_results.items():
        gt_boxes = img_result['ground_truth_boxes']
        matches = img_result['matches']
        unmatched_preds = img_result['unmatched_preds']
        unmatched_gts = img_result['unmatched_gts']
        max_confidence = img_result['max_confidence']
        
        # Accumulate detection-level statistics
        total_gt_boxes += len(gt_boxes)
        total_predictions += img_result['num_predictions']
        
        # Count true positive detections (good IoU matches)
        good_matches = [m for m in matches if m['iou'] >= iou_threshold and m['confidence'] >= conf_threshold]
        tp_detections += len(good_matches)
        
        # Count false positive detections (unmatched predictions + poor IoU matches)
        poor_matches = [m for m in matches if m['iou'] < iou_threshold or m['confidence'] < conf_threshold]
        fp_detections += len(unmatched_preds) + len(poor_matches)
        
        # Count false negative detections (unmatched ground truth)
        fn_detections += len(unmatched_gts)
        
        # Collect IoUs and confidences
        for match in matches:
            all_ious.append(match['iou'])
            matched_confidences.append(match['confidence'])
        
        is_abnormal = 'abnormal' in img_id
        has_good_detection = len(good_matches) > 0
        has_any_detection = max_confidence >= conf_threshold
        if is_abnormal and has_good_detection:
            tp_images += 1
        elif is_abnormal and (not has_any_detection or not has_good_detection):
            fn_images += 1
        elif not is_abnormal and has_any_detection:
            fp_images += 1
        else:
            tn_images += 1
    
    # Calculate detection-level metrics
    detection_precision = tp_detections / (tp_detections + fp_detections) if (tp_detections + fp_detections) > 0 else 0
    detection_recall = tp_detections / (tp_detections + fn_detections) if (tp_detections + fn_detections) > 0 else 0
    detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0
    
    # Calculate image-level metrics
    total_images = tp_images + fp_images + tn_images + fn_images
    image_accuracy = (tp_images + tn_images) / total_images if total_images > 0 else 0
    image_precision = tp_images / (tp_images + fp_images) if (tp_images + fp_images) > 0 else 0
    image_recall = tp_images / (tp_images + fn_images) if (tp_images + fn_images) > 0 else 0
    image_f1 = 2 * (image_precision * image_recall) / (image_precision + image_recall) if (image_precision + image_recall) > 0 else 0
    image_specificity = tn_images / (tn_images + fp_images) if (tn_images + fp_images) > 0 else 0
    
    # Calculate IoU statistics
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    median_iou = np.median(all_ious) if all_ious else 0.0
    mean_confidence = np.mean(matched_confidences) if matched_confidences else 0.0
    
    # Copy and classify test images with IoU analysis
    classification_results = copy_test_images_and_classify_with_iou(
        test_data=test_data,
        evaluation_results=evaluation_results,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    # Print comprehensive results
    print(f"\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDATASET STATISTICS:")
    print(f"  Total test images: {total_images}")
    print(f"  Total ground truth boxes: {total_gt_boxes}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Total matches found: {len(all_ious)}")
    print(f"  Good matches (IoU≥{iou_threshold}): {tp_detections}")
    
    print(f"\nIoU STATISTICS:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Median IoU: {median_iou:.4f}")
    print(f"  Mean Confidence (matched): {mean_confidence:.4f}")
    print(f"  IoU Threshold: {iou_threshold}")
    print(f"  Confidence Threshold: {conf_threshold}")
    
    print(f"\nDETECTION-LEVEL METRICS (Box-level evaluation):")
    print(f"  True Positive Detections:  {tp_detections}")
    print(f"  False Positive Detections: {fp_detections}")
    print(f"  False Negative Detections: {fn_detections}")
    print(f"  Precision@IoU{iou_threshold}: {detection_precision:.4f} ({detection_precision*100:.2f}%)")
    print(f"  Recall@IoU{iou_threshold}:    {detection_recall:.4f} ({detection_recall*100:.2f}%)")
    print(f"  F1-Score@IoU{iou_threshold}:  {detection_f1:.4f} ({detection_f1*100:.2f}%)")
    
    print(f"\nIMAGE-LEVEL METRICS:")
    print(f"  TP: {tp_images}, FP: {fp_images}, TN: {tn_images}, FN: {fn_images}")
    print(f"  Accuracy@IoU{iou_threshold}:    {image_accuracy:.4f} ({image_accuracy*100:.2f}%)")
    print(f"  Precision@IoU{iou_threshold}:   {image_precision:.4f} ({image_precision*100:.2f}%)")
    print(f"  Recall@IoU{iou_threshold}:      {image_recall:.4f} ({image_recall*100:.2f}%)")
    print(f"  F1-Score@IoU{iou_threshold}:    {image_f1:.4f} ({image_f1*100:.2f}%)")
    print(f"  Specificity@IoU{iou_threshold}: {image_specificity:.4f} ({image_specificity*100:.2f}%)")
    
    print(f"\nCOCO EVALUATION METRICS:")
    for metric_name, value in coco_metrics.items():
        print(f"  {metric_name}: {value:.4f} ({value*100:.2f}%)")
    
    # Prepare comprehensive results
    model_results = {
        'model_type': 'comprehensive_iou_evaluation',
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'test_split': test_data.get('test_split', 'unknown'),
        'total_test_images': total_images,
        'total_gt_boxes': total_gt_boxes,
        'total_predictions': total_predictions,
        
        # Detection-level metrics
        'detection_tp': tp_detections,
        'detection_fp': fp_detections,
        'detection_fn': fn_detections,
        'detection_precision': detection_precision,
        'detection_recall': detection_recall,
        'detection_f1': detection_f1,
        
        # Image-level metrics
        'image_tp': tp_images,
        'image_fp': fp_images,
        'image_tn': tn_images,
        'image_fn': fn_images,
        'image_accuracy': image_accuracy,
        'image_precision': image_precision,
        'image_recall': image_recall,
        'image_f1': image_f1,
        'image_specificity': image_specificity,
        
        # IoU statistics
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'mean_confidence': mean_confidence,
        
        # COCO metrics
        **coco_metrics,
        
        'output_dir': output_dir,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save comprehensive results
    results_df = pd.DataFrame([model_results])
    results_path = os.path.join(output_dir, 'comprehensive_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save detailed IoU analysis
    iou_analysis = {
        'all_ious': all_ious,
        'matched_confidences': matched_confidences,
        'evaluation_summary': model_results
    }
    
    import json
    iou_analysis_path = os.path.join(output_dir, 'iou_analysis.json')
    with open(iou_analysis_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_analysis = {
            'all_ious': [float(x) for x in all_ious],
            'matched_confidences': [float(x) for x in matched_confidences],
            'evaluation_summary': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                 for k, v in model_results.items()}
        }
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - Comprehensive metrics: {results_path}")
    print(f"  - IoU analysis: {iou_analysis_path}")
    print(f"  - Test images organized in: {classification_results['test_dir']}")
    print(f"  - Enhanced visualizations: {classification_results['test_dir']}/visualizations")
    
    return model_results

def get_ground_truth_boxes(img_path: str, annotations_df: pd.DataFrame, image_size: tuple):
    """Extract ground truth bounding boxes for an image"""
    path_col = _path_column(annotations_df)
    image_annotations = annotations_df[annotations_df[path_col] == img_path]
    gt_boxes = []
    
    width, height = image_size
    
    for _, row in image_annotations.iterrows():
        bbox_str = row['bbox']
        if bbox_str and bbox_str != '':
            try:
                bbox_list = ast.literal_eval(bbox_str)
                for bbox_dict in bbox_list:
                    # Extract coordinates (in percentages)
                    x_percent = bbox_dict['x']
                    y_percent = bbox_dict['y'] 
                    width_percent = bbox_dict['width']
                    height_percent = bbox_dict['height']
                    
                    # Convert to absolute coordinates
                    x = (x_percent / 100) * width
                    y = (y_percent / 100) * height
                    w = (width_percent / 100) * width
                    h = (height_percent / 100) * height
                    
                    # Convert to pascal VOC format (xmin, ymin, xmax, ymax)
                    xmin = max(0, min(x, width - 1))
                    ymin = max(0, min(y, height - 1))
                    xmax = max(xmin + 1, min(x + w, width))
                    ymax = max(ymin + 1, min(y + h, height))
                    
                    gt_boxes.append([xmin, ymin, xmax, ymax])
                    
            except Exception as e:
                print(f"Error parsing bbox for image {img_path}: {e}")
                continue
    
    return np.array(gt_boxes) if gt_boxes else np.array([]).reshape(0, 4)

def split_dataset_for_testing(data_dir: str, csv_path: str, abnormal_dir_name: str = None,
                              test_size: int = 200, normal_limit: int = None, seed: int = 42):
    """Split dataset into test sets with fixed seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if abnormal_dir_name:
        abnormal_dir = os.path.join(data_dir, abnormal_dir_name)
        normal_dir = os.path.join(data_dir, "normal")
        if not os.path.exists(normal_dir):
            normal_dir = abnormal_dir
    else:
        abnormal_dir = os.path.abspath(data_dir)
        normal_dir = abnormal_dir
    annotations_df = pd.read_csv(csv_path)
    path_col = _path_column(annotations_df)
    if 'label' in annotations_df.columns:
        labels = annotations_df['label'].astype(str).str.strip().str.lower()
        abnormal_df = annotations_df[labels == 'abnormal']
        normal_df = annotations_df[labels == 'normal']
        abnormal_image_list = abnormal_df[path_col].unique().tolist()
        csv_normal_image_list = normal_df[path_col].unique().tolist() if len(normal_df) else []
    else:
        abnormal_image_list = annotations_df[path_col].unique().tolist()
        csv_normal_image_list = []
    existing_abnormal = []
    for img_path in abnormal_image_list:
        candidates = [img_path] if os.path.isabs(img_path) else []
        candidates += [
            os.path.join(abnormal_dir, img_path.lstrip('/')),
            os.path.join(abnormal_dir, os.path.basename(img_path)),
            os.path.join(abnormal_dir, img_path.replace('/', '_'))
        ]
        if next((p for p in candidates if os.path.exists(p)), None):
            existing_abnormal.append(img_path)
    all_normal = list(csv_normal_image_list)
    need_walk = os.path.exists(normal_dir) and (
        normal_dir != abnormal_dir or not csv_normal_image_list
    )
    if need_walk:
        all_normal_set = set(all_normal)
        for root, _, files in os.walk(normal_dir):
            for f in files:
                if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                    rp = os.path.relpath(os.path.join(root, f), normal_dir)
                    if rp not in all_normal_set:
                        all_normal_set.add(rp)
                        all_normal.append(rp)
    if normal_limit and len(all_normal) > normal_limit:
        random.shuffle(all_normal)
        all_normal = all_normal[:normal_limit]
    random.shuffle(existing_abnormal)
    random.shuffle(all_normal)
    test_abnormal = existing_abnormal[:min(test_size, len(existing_abnormal))]
    test_normal = all_normal[:min(test_size, len(all_normal))]
    return {
        'abnormal': test_abnormal,
        'normal': test_normal,
        'abnormal_dir': abnormal_dir,
        'normal_dir': normal_dir,
        'annotations_df': annotations_df
    }

def copy_test_images_and_classify_with_iou(test_data: dict, evaluation_results: dict, 
                                          output_dir: str, conf_threshold: float = 0.3,
                                          iou_threshold: float = 0.5):
    """
    Copy test images to organized directories based on classification results with IoU analysis
    """
    # Create test directory structure
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create classification directories
    class_dirs = {
        'tp': os.path.join(test_dir, 'true_positive'),
        'fp': os.path.join(test_dir, 'false_positive'),
        'tn': os.path.join(test_dir, 'true_negative'),
        'fn': os.path.join(test_dir, 'false_negative'),
        'visualizations': os.path.join(test_dir, 'visualizations')
    }
    
    for dir_path in class_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Classification counters
    classification_counts = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    
    for img_path in test_data['abnormal']:
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        candidates.append(os.path.join(test_data['abnormal_dir'], img_path.lstrip('/')))
        candidates.append(os.path.join(test_data['abnormal_dir'], os.path.basename(img_path)))
        candidates.append(os.path.join(test_data['abnormal_dir'], img_path.replace('/', '_')))
        source_path = next((p for p in candidates if os.path.exists(p)), None)
        if source_path is None:
            continue
        img_id = f"abnormal_{hash(img_path) % 100000}"
        
        # Get evaluation results for this image
        img_result = evaluation_results.get(img_id, {})
        matches = img_result.get('matches', [])
        mean_iou = img_result.get('mean_iou', 0.0)
        max_confidence = img_result.get('max_confidence', 0.0)
        
        # Classify based on IoU-aware detection
        if len(matches) > 0 and max_confidence >= conf_threshold:
            # Check if any match has IoU >= threshold
            good_matches = [m for m in matches if m['iou'] >= iou_threshold]
            if good_matches:
                dest_dir = class_dirs['tp']
                classification = 'TP'
            else:
                dest_dir = class_dirs['fn'] 
                classification = 'FN'  # Low IoU detection = missed
        else:
            dest_dir = class_dirs['fn']
            classification = 'FN'
        
        # Copy image with detailed filename
        dest_filename = f"{classification}_mIoU{mean_iou:.3f}_conf{max_confidence:.3f}_{img_id}_{os.path.basename(source_path)}"
        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copy2(source_path, dest_path)
        classification_counts[classification.lower()] += 1
    
    # Process normal test images
    for img_path in test_data['normal']:
        source_path = os.path.join(test_data['normal_dir'], img_path)
        
        if not os.path.exists(source_path):
            continue
            
        # Create unique identifier for this image
        img_id = f"normal_{hash(img_path) % 100000}"
        
        # Get evaluation results for this image  
        img_result = evaluation_results.get(img_id, {})
        max_confidence = img_result.get('max_confidence', 0.0)
        
        # Classify: FP if detected above threshold, TN if correctly classified as normal
        if max_confidence >= conf_threshold:
            dest_dir = class_dirs['fp']
            classification = 'FP'
        else:
            dest_dir = class_dirs['tn']
            classification = 'TN'
        
        # Copy image with detailed filename
        dest_filename = f"{classification}_conf{max_confidence:.3f}_{img_id}_{os.path.basename(source_path)}"
        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copy2(source_path, dest_path)
        classification_counts[classification.lower()] += 1
    
    print(f"\nClassification Summary (conf_threshold={conf_threshold}, iou_threshold={iou_threshold}):")
    print(f"  True Positives (TP):  {classification_counts['tp']}")
    print(f"  False Positives (FP): {classification_counts['fp']}")
    print(f"  True Negatives (TN):  {classification_counts['tn']}")
    print(f"  False Negatives (FN): {classification_counts['fn']}")
    
    return {
        'tp': classification_counts['tp'],
        'fp': classification_counts['fp'],
        'tn': classification_counts['tn'],
        'fn': classification_counts['fn'],
        'test_dir': test_dir
    }

def create_enhanced_visualization(image_path: str, img_result: dict, img_id: str,
                                is_abnormal: bool, gt_boxes: np.ndarray, output_dir: str,
                                conf_threshold: float = 0.3, iou_threshold: float = 0.5):
    """Create enhanced visualization with IoU and confidence information"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return
        
        img_vis = img.copy()
        height, width = img_vis.shape[:2]
        
        # Get results
        predictions = img_result.get('predictions', {})
        matches = img_result.get('matches', [])
        unmatched_preds = img_result.get('unmatched_preds', [])
        mean_iou = img_result.get('mean_iou', 0.0)
        max_confidence = img_result.get('max_confidence', 0.0)
        
        pred_boxes = predictions.get('boxes', [])
        pred_scores = predictions.get('scores', [])
        
        # Draw ground truth boxes in blue
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in gt_box]
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for GT
            cv2.putText(img_vis, 'GT', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw matched predictions in green with IoU
        for match in matches:
            pred_idx = match['pred_idx']
            iou = match['iou']
            confidence = match['confidence']
            pred_box = match['pred_box']
            
            if confidence >= conf_threshold:
                x1, y1, x2, y2 = [int(coord) for coord in pred_box]
                
                # Color based on IoU quality
                if iou >= iou_threshold:
                    color = (0, 255, 0)  # Green for good IoU
                else:
                    color = (0, 255, 255)  # Yellow for low IoU
                
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_vis, f'IoU:{iou:.2f} C:{confidence:.2f}', 
                          (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw unmatched predictions in red
        for pred_idx in unmatched_preds:
            if pred_idx < len(pred_boxes) and pred_idx < len(pred_scores):
                if pred_scores[pred_idx] >= conf_threshold:
                    pred_box = pred_boxes[pred_idx]
                    confidence = pred_scores[pred_idx]
                    x1, y1, x2, y2 = [int(coord) for coord in pred_box]
                    
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for FP
                    cv2.putText(img_vis, f'FP C:{confidence:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        predicted_abnormal = max_confidence >= conf_threshold
        good_matches = [m for m in matches if m['iou'] >= iou_threshold and m['confidence'] >= conf_threshold]
        if is_abnormal and len(good_matches) > 0:
            classification = "TP"
            class_color = (0, 255, 0)
        elif is_abnormal and (not predicted_abnormal or len(good_matches) == 0):
            classification = "FN"
            class_color = (0, 0, 255)
        elif not is_abnormal and predicted_abnormal:
            classification = "FP"
            class_color = (0, 165, 255)
        else:
            classification = "TN"
            class_color = (255, 0, 0)
        header_text = f"Mean IoU: {mean_iou:.3f} | Max Conf: {max_confidence:.3f}"
        cv2.putText(img_vis, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_vis, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        classification_text = f"{classification} - GT: {'Abnormal' if is_abnormal else 'Normal'} | Matches: {len(matches)} | Good: {len(good_matches)}"
        cv2.putText(img_vis, classification_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)
        
        # Add legend
        legend_y = height - 100
        cv2.putText(img_vis, 'Legend: GT=Blue, Good IoU=Green, Low IoU=Yellow, FP=Red', 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img_vis, 'Legend: GT=Blue, Good IoU=Green, Low IoU=Yellow, FP=Red', 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save visualization
        vis_dir = os.path.join(output_dir, 'test', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        vis_filename = f"{classification}_mIoU{mean_iou:.3f}_conf{max_confidence:.3f}_{img_id}_{os.path.basename(image_path)}"
        vis_path = os.path.join(vis_dir, vis_filename)
        cv2.imwrite(vis_path, img_vis)
        
    except Exception as e:
        print(f"Error creating visualization for {image_path}: {e}")

# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

if __name__ == "__main__":
    args = parse_arguments()
    run_evaluation = args.run_evaluation and not args.skip_evaluation
    if args.num_workers is None:
        args.num_workers = min(8, os.cpu_count())
    
    print("=" * 80)
    print("PATHOLOGY DETECTION TRAINING - EfficientDet")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Abnormal folder: {args.abnormal_dir or '(same as data_dir)'}")
    print(f"CSV path: {args.csv_path}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Train split: {args.train_split}")
    print(f"Test split: {args.test_split}")
    print(f"Validation split: {args.val_split}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Random seed: {args.eval_seed}")
    print(f"Run evaluation: {run_evaluation}")
    if run_evaluation:
        print(f"  Normal image limit: {args.normal_limit}")
        print(f"  Confidence threshold: {args.eval_conf_threshold}")
        print(f"  IoU threshold: {args.eval_iou_threshold}")
    
    # Get image size from model configuration
    try:
        config = get_efficientdet_config(args.model_name)
        image_size = config.image_size[0]
        print(f"Image size (from model config): {image_size}")
    except Exception as e:
        print(f"Error getting model config: {e}")
        print("Please check if the model name is valid.")
        exit(1)
    
    print("=" * 80)
    
    try:
        model, trainer, evaluation_results = train_detection(
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            abnormal_dir_name=args.abnormal_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            train_split=args.train_split,
            test_split=args.test_split,
            val_split=args.val_split,
            num_workers=args.num_workers,
            run_evaluation=run_evaluation,
            normal_limit=args.normal_limit,
            eval_conf_threshold=args.eval_conf_threshold,
            eval_iou_threshold=args.eval_iou_threshold,
            eval_seed=args.eval_seed
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        if run_evaluation:
            if evaluation_results:
                print("EVALUATION COMPLETED SUCCESSFULLY!")
                print(f"Evaluated {len(evaluation_results)} model(s)")
            else:
                print("EVALUATION COMPLETED WITH ERRORS!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        exit(1) 