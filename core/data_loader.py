import logging
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from transformers import AutoImageProcessor
from typing import Tuple, Optional, Dict, Any, List
import cv2

logger = logging.getLogger(__name__)


def collate_fn_with_paths(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors; keep file_path as a list (default_collate cannot batch strings)."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    out: Dict[str, Any] = {"pixel_values": pixel_values, "labels": labels}
    if batch and "subtype_targets" in batch[0]:
        out["subtype_targets"] = torch.stack([b["subtype_targets"] for b in batch])
    if batch and "file_path" in batch[0]:
        out["file_path"] = [b["file_path"] for b in batch]
    return out


class ChestXrayDataset(Dataset):
    """Dataset class for chest X-ray images."""
    
    def __init__(self, 
                 data_dir: str,
                 csv_file: Optional[str] = None,
                 transform: Optional[A.Compose] = None,
                 image_processor: Optional[AutoImageProcessor] = None,
                 data: Optional[pd.DataFrame] = None,
                 return_file_path: bool = False,
                 subtype_columns: Optional[List[str]] = None):
        """
        Args:
            data_dir: Base directory for relative paths in CSV
            csv_file: CSV file with image paths and labels
            transform: Albumentations transform pipeline
            image_processor: HuggingFace image processor
            data: Optional in-memory DataFrame (e.g. inference with file_path + label)
            return_file_path: If True, __getitem__ includes 'file_path' (use collate_fn_with_paths in DataLoader)
            subtype_columns: Optional list of binary subtype-label columns. Those present
                in the data are returned per-item as a float 'subtype_targets' tensor.
        """
        self.data_dir = data_dir or ""
        self.transform = transform
        self.image_processor = image_processor
        self.return_file_path = return_file_path
        self._requested_subtype_columns = list(subtype_columns or [])
        
        if data is not None:
            self.data = data.copy()
        elif csv_file and os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            self.data = self._load_from_directory()
        
        if "file_path" in self.data.columns:
            self.path_column = "file_path"
        elif "image_path" in self.data.columns:
            self.path_column = "image_path"
        elif "path" in self.data.columns:
            self.path_column = "path"
        else:
            raise ValueError("CSV must contain one of: file_path, image_path, path")
        self.label_map = {"normal": 0, "abnormal": 1}
        
        if "label" not in self.data.columns:
            self.data["label"] = 0
            self._label_is_numeric = True
        else:
            try:
                self.data["label"] = self.data["label"].astype(int)
                self._label_is_numeric = True
            except (ValueError, TypeError):
                self._label_is_numeric = False

        n_before = len(self.data)
        self.data = self._filter_existing_files(self.data).reset_index(drop=True)
        n_dropped = n_before - len(self.data)
        if n_dropped:
            logger.warning(
                "Dropped %d rows with missing image files (kept %d)",
                n_dropped,
                len(self.data),
            )
        if len(self.data) == 0:
            raise ValueError(
                "No valid image files found. Check data_dir and CSV paths "
                f"(data_dir={self.data_dir!r}, path_column={self.path_column!r})"
            )

        # Resolve which requested subtype columns are actually present.
        self.subtype_columns = [
            c for c in self._requested_subtype_columns if c in self.data.columns
        ]
        if self.subtype_columns:
            self._subtype_matrix = (
                self.data[self.subtype_columns].fillna(0).astype("float32").values
            )
        else:
            self._subtype_matrix = None

    def resolve_image_path(self, raw_path: str) -> str:
        if not os.path.isabs(str(raw_path)) and self.data_dir:
            return os.path.normpath(os.path.join(self.data_dir, str(raw_path)))
        return os.path.normpath(str(raw_path))

    def _filter_existing_files(self, df: pd.DataFrame) -> pd.DataFrame:
        paths = df[self.path_column].map(self.resolve_image_path)
        exists = paths.map(os.path.isfile)
        return df.loc[exists].copy()

    def _load_from_directory(self) -> pd.DataFrame:
        """Load data from directory structure (normal/abnormal folders)."""
        data = []
        
        for label in ['normal', 'abnormal']:
            label_dir = os.path.join(self.data_dir, label)
            if os.path.exists(label_dir):
                for img_file in os.listdir(label_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                        data.append({
                            'image_path': os.path.join(label_dir, img_file),
                            'label': label
                        })
        
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        image_path = self.resolve_image_path(row[self.path_column])
        image = self._load_image(image_path)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Process with HuggingFace processor
        if self.image_processor:
            # Convert numpy array to PIL Image for consistent processing
            if isinstance(image, np.ndarray):
                # Ensure image is in the right format for PIL
                if image.dtype == np.float32:
                    image = (image * 255.0).astype(np.uint8)
                elif image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                
                # Convert to PIL Image
                image = Image.fromarray(image)
            
            # Process with HuggingFace processor
            processed = self.image_processor(image, return_tensors="pt")
            pixel_values = processed['pixel_values'].squeeze(0)
        else:
            # Fallback: convert to tensor manually
            if isinstance(image, np.ndarray):
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                pixel_values = torch.tensor(image).permute(2, 0, 1)
            else:
                pixel_values = image
        
        if self._label_is_numeric:
            label = int(row["label"])
        else:
            raw = row["label"]
            if pd.isna(raw):
                label = 0
            else:
                label = self.label_map.get(
                    raw, self.label_map.get(str(raw).strip().lower(), 0)
                )
        
        out: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }
        if self._subtype_matrix is not None:
            out["subtype_targets"] = torch.from_numpy(self._subtype_matrix[idx])
        if self.return_file_path:
            out["file_path"] = image_path
        return out
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_path.lower().endswith(".dcm"):
            import pydicom

            ds = pydicom.dcmread(image_path)
            image = ds.pixel_array
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(
                np.uint8
            )
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                try:
                    with Image.open(image_path) as pil_img:
                        image = np.array(pil_img.convert("L"))
                except Exception as e:
                    raise OSError(f"Could not read image: {image_path}") from e

        if image is None:
            raise OSError(f"Could not read image: {image_path}")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

def get_transforms(image_size: int = 224, is_training: bool = True,
                   use_clahe: bool = False, use_letterbox: bool = False) -> A.Compose:
    """Get augmentation pipeline. use_clahe prepends CLAHE contrast enhancement.

    use_letterbox: aspect-preserving resize (long side -> image_size) + zero-pad to a
    square, instead of A.Resize which squashes to square. Important for tall/narrow
    spine crops where square-resize distorts vertebra proportions.
    """
    clahe = [A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)] if use_clahe else []
    if use_letterbox:
        resize = [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0),
        ]
    else:
        resize = [A.Resize(image_size, image_size)]

    if is_training:
        return A.Compose(clahe + resize + [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(noise_scale_factor=0.1, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            # Remove normalization and tensor conversion - let HuggingFace processor handle it
        ])
    else:
        return A.Compose(clahe + resize + [
            # Remove normalization and tensor conversion - let HuggingFace processor handle it
        ])


def build_image_processor(config) -> "AutoImageProcessor":
    """Build the HF image processor, optionally forcing a non-native square input size.

    rad-dino-maira-2 defaults to 518px (resize shortest-edge + center-crop). Setting
    config.processor_image_size overrides both so the encoder actually sees higher-res
    images (paired with interpolate_pos_encoding=True in the model forward).
    """
    proc = AutoImageProcessor.from_pretrained(config.model_name, use_fast=True)
    n = getattr(config, "processor_image_size", None)
    if n:
        n = int(n)
        if hasattr(proc, "size") and isinstance(proc.size, dict):
            if "shortest_edge" in proc.size:
                proc.size = {"shortest_edge": n}
            else:
                proc.size = {"height": n, "width": n}
        if hasattr(proc, "crop_size") and isinstance(proc.crop_size, dict):
            proc.crop_size = {"height": n, "width": n}
        logger.info("Image processor forced to %dpx (override of native default)", n)
    return proc


def _binary_y_from_dataset(dataset: ChestXrayDataset) -> np.ndarray:
    col = dataset.data["label"]
    if dataset._label_is_numeric:
        return col.astype(int).values
    return col.map(
        lambda x: 1 if str(x).strip().lower() == "abnormal" else 0
    ).values


def get_bce_pos_weight_from_dataset(dataset: ChestXrayDataset) -> torch.Tensor:
    y = _binary_y_from_dataset(dataset)
    n0 = max(1, int((y == 0).sum()))
    n1 = max(1, int((y == 1).sum()))
    return torch.tensor([n0 / n1], dtype=torch.float32)


def get_class_weights_from_dataset(dataset: ChestXrayDataset) -> torch.Tensor:
    y = _binary_y_from_dataset(dataset)
    n0 = max(1, int((y == 0).sum()))
    n1 = max(1, int((y == 1).sum()))
    total = n0 + n1
    return torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)


def build_subtype_sampler_weights(
    dataset: ChestXrayDataset, alpha: float = 0.5
) -> Optional[torch.Tensor]:
    """Per-sample weights that oversample images carrying the rarer target subtypes.

    For each subtype column, rarity = mean_count / count_subtype. A sample's weight
    is the max rarity over the subtypes it has (so multi-finding images aren't blown
    up multiplicatively), raised to ``alpha`` to temper aggressiveness. Samples with
    none of the target subtypes (e.g. normal) get weight 1.0. Returns None if the
    dataset has no subtype columns.
    """
    if getattr(dataset, "_subtype_matrix", None) is None:
        return None
    mat = dataset._subtype_matrix  # (N, K) float 0/1
    counts = mat.sum(axis=0)
    counts = np.maximum(counts, 1.0)
    mean_count = float(counts.mean())
    rarity = (mean_count / counts) ** float(alpha)  # (K,)
    # weight = max rarity among present subtypes, floored at 1.0 so the sampler
    # only ever boosts rarer subtypes and never samples any image (normal or a
    # common subtype) below baseline.
    present_rarity = mat * rarity[None, :]
    w = present_rarity.max(axis=1)
    w = np.maximum(w, 1.0).astype("float64")
    return torch.as_tensor(w, dtype=torch.double)


def _create_data_loaders_from_data_csv(
    config, batch_size: Optional[int]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if not config.data_csv or not os.path.isfile(config.data_csv):
        raise FileNotFoundError(f"data_csv not found: {getattr(config, 'data_csv', None)}")

    image_processor = build_image_processor(config)
    df = pd.read_csv(config.data_csv)
    if df.empty:
        raise ValueError(f"CSV is empty: {config.data_csv}")

    seed = int(getattr(config, "seed", 42))
    group_col = getattr(config, "group_split_column", None)
    if group_col and group_col in df.columns:
        # Group-aware split: assign whole groups (e.g. all images of a study_iuid) to
        # one split so no study leaks across train/val/test. Deterministic by seed.
        groups = df[group_col].astype(str)
        uniq = (
            pd.Series(groups.unique())
            .sample(frac=1.0, random_state=seed)
            .tolist()
        )
        g = len(uniq)
        g_train = int(g * config.train_split)
        g_val = int(g * config.val_split)
        train_g = set(uniq[:g_train])
        val_g = set(uniq[g_train : g_train + g_val])
        train_df = df.loc[groups.isin(train_g)].copy()
        val_df = df.loc[groups.isin(val_g)].copy()
        test_df = df.loc[~groups.isin(train_g | val_g)].copy()
        logger.info(
            "Group-aware split on %r: %d groups -> train=%d val=%d test=%d images "
            "(train=%d val=%d test=%d groups)",
            group_col, g, len(train_df), len(val_df), len(test_df),
            len(train_g), len(val_g), g - len(train_g) - len(val_g),
        )
    else:
        if group_col:
            logger.warning(
                "group_split_column %r not in CSV; falling back to row-wise split.",
                group_col,
            )
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_train = int(n * config.train_split)
        n_val = int(n * config.val_split)
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train : n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val :].copy()
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Invalid split n={n}: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    base_dir = config.data_dir or ""
    bs = batch_size if batch_size is not None else config.batch_size
    nw = min(16, config.num_workers)
    subtype_columns = list(getattr(config, "subtype_names", []) or [])
    return_paths = bool(getattr(config, "return_file_path_eval", False))

    train_dataset = ChestXrayDataset(
        data_dir=base_dir,
        csv_file=None,
        data=train_df,
        transform=get_transforms(config.image_size, is_training=True, use_clahe=getattr(config,'use_clahe',False), use_letterbox=getattr(config,'use_letterbox',False)),
        image_processor=image_processor,
        subtype_columns=subtype_columns,
    )
    val_dataset = ChestXrayDataset(
        data_dir=base_dir,
        csv_file=None,
        data=val_df,
        transform=get_transforms(config.image_size, is_training=False, use_clahe=getattr(config,'use_clahe',False), use_letterbox=getattr(config,'use_letterbox',False)),
        image_processor=image_processor,
        subtype_columns=subtype_columns,
    )
    test_dataset = ChestXrayDataset(
        data_dir=base_dir,
        csv_file=None,
        data=test_df,
        transform=get_transforms(config.image_size, is_training=False, use_clahe=getattr(config,'use_clahe',False), use_letterbox=getattr(config,'use_letterbox',False)),
        image_processor=image_processor,
        subtype_columns=subtype_columns,
        return_file_path=return_paths,
    )

    # Subtype-weighted sampler (train only). Mutually exclusive with shuffle.
    train_sampler = None
    # Per-example cost-sensitive sampler: oversample by an explicit weight column
    # (equivalent in expectation to per-example loss weighting). Takes priority.
    weight_col = getattr(config, "weight_column", None)
    if weight_col and weight_col in getattr(train_dataset, "data", train_df).columns:
        wcol = (train_dataset.data[weight_col].fillna(1.0)
                .astype("float32").clip(lower=1e-6).values)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(wcol, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        logger.warning("Per-example weight sampler from '%s' (mean=%.2f max=%.2f, n=%d)",
                       weight_col, float(wcol.mean()), float(wcol.max()), len(wcol))
    if train_sampler is None and bool(getattr(config, "use_subtype_sampler", False)):
        train_sampler = build_subtype_sampler_weights(
            train_dataset, alpha=float(getattr(config, "sampler_alpha", 0.5))
        )
        if train_sampler is not None:
            train_sampler = WeightedRandomSampler(
                weights=train_sampler,
                num_samples=len(train_dataset),
                replacement=True,
            )
            logger.info("Using subtype-weighted sampler (alpha=%.2f)",
                        float(getattr(config, "sampler_alpha", 0.5)))

    # Collate must pass subtype_targets/file_path through when present.
    use_collate = (subtype_columns or return_paths)
    collate = collate_fn_with_paths if use_collate else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=False,
        collate_fn=collate,
    )
    return train_loader, val_loader, test_loader


def create_data_loaders(
    config, batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val / test loaders. Optional batch_size overrides config.batch_size."""
    bs = batch_size if batch_size is not None else config.batch_size

    if getattr(config, "data_csv", None):
        return _create_data_loaders_from_data_csv(config, batch_size)

    image_processor = build_image_processor(config)

    train_dataset = ChestXrayDataset(
        data_dir=os.path.join(config.data_dir, "train"),
        csv_file=config.train_csv,
        transform=get_transforms(config.image_size, is_training=True, use_clahe=getattr(config,'use_clahe',False), use_letterbox=getattr(config,'use_letterbox',False)),
        image_processor=image_processor,
    )
    val_dataset = ChestXrayDataset(
        data_dir=os.path.join(config.data_dir, "val"),
        csv_file=config.val_csv,
        transform=get_transforms(config.image_size, is_training=False, use_clahe=getattr(config,'use_clahe',False), use_letterbox=getattr(config,'use_letterbox',False)),
        image_processor=image_processor,
    )
    test_dataset = ChestXrayDataset(
        data_dir=os.path.join(config.data_dir, "test"),
        csv_file=config.test_csv,
        transform=get_transforms(config.image_size, is_training=False, use_clahe=getattr(config,'use_clahe',False), use_letterbox=getattr(config,'use_letterbox',False)),
        image_processor=image_processor,
    )

    nw = min(16, config.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader

def get_class_weights(data_dir: str) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset."""
    normal_count = len(os.listdir(os.path.join(data_dir, 'normal')))
    abnormal_count = len(os.listdir(os.path.join(data_dir, 'abnormal')))
    
    total = normal_count + abnormal_count
    weights = torch.tensor([total / (2 * normal_count), total / (2 * abnormal_count)])
    
    return weights 