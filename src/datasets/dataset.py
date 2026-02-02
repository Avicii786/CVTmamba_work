import os
import random
import numpy as np
import torch
from skimage import io
from torch.utils import data
import torchvision.transforms.functional as TF
from PIL import Image

# Import the centralized normalization logic
from src.utils import normalization_utils as norm_utils

# ==============================================================================
# DATASET CONFIGURATIONS
# ==============================================================================

# --- SECOND Dataset Config ---
ST_NUM_CLASSES = 7
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
ST_COLORMAP = [
    [255, 255, 255], # Unchanged
    [0, 0, 255],     # Water
    [128, 128, 128], # Ground
    [0, 128, 0],     # Low Vegetation
    [0, 255, 0],     # Tree
    [128, 0, 0],     # Building
    [255, 0, 0]      # Sports Field
]

# --- LandsatSCD Dataset Config ---
LANDSAT_NUM_CLASSES = 5
LANDSAT_CLASSES = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
LANDSAT_COLORMAP = [
    [255, 255, 255], # 0: No change
    [0, 155, 0],     # 1: Farmland
    [255, 165, 0],   # 2: Desert
    [230, 30, 100],  # 3: Building
    [0, 170, 240]    # 4: Water
]

def build_colormap_lookup(colormap):
    """
    Builds a lookup table for RGB -> Class Index conversion.
    """
    lookup = np.zeros(256 ** 3, dtype=np.uint8)
    for i, cm in enumerate(colormap):
        # Calculate linear index for RGB triplet
        idx = (cm[0] * 256 + cm[1]) * 256 + cm[2]
        lookup[idx] = i
    return lookup

# Pre-calculate lookup tables
st_colormap2label = build_colormap_lookup(ST_COLORMAP)
landsat_colormap2label = build_colormap_lookup(LANDSAT_COLORMAP)

def Color2Index(ColorLabel, lookup_table, num_classes):
    """
    Converts an RGB label image (H, W, 3) to a 2D index mask (H, W)
    using a specific lookup table.
    """
    if len(ColorLabel.shape) == 2: # Already grayscale/index
        return ColorLabel
        
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    
    IndexMap = lookup_table[idx]
    
    # Clip any out-of-bounds indices (e.g. noise colors) to 'unchanged' (0) 
    IndexMap = IndexMap * (IndexMap < num_classes)
    
    return IndexMap.astype(np.int64)

# ==============================================================================
# Main Dataset Class
# ==============================================================================

class SCDDataset(data.Dataset):
    """
    Universal DataLoader for Semantic Change Detection Datasets.
    """
    def __init__(self, root, mode, dataset_name='SECOND', random_flip=False, random_swap=False):
        self.root = root
        self.mode = mode
        self.dataset_name = dataset_name
        self.random_flip = random_flip
        self.random_swap = random_swap
        
        # 1. Select Configuration based on dataset name
        if 'second' in dataset_name.lower():
            self.lookup_table = st_colormap2label
            self.num_classes = ST_NUM_CLASSES
            self.class_names = ST_CLASSES
        elif 'landsat' in dataset_name.lower():
            self.lookup_table = landsat_colormap2label
            self.num_classes = LANDSAT_NUM_CLASSES
            self.class_names = LANDSAT_CLASSES
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'SECOND' or 'LandsatSCD'.")

        # 2. Setup paths
        base_path = os.path.join(root, mode)
        
        self.dir_A = os.path.join(base_path, 'A')
        self.dir_B = os.path.join(base_path, 'B')
        self.dir_sem1 = os.path.join(base_path, 'labelA_rgb')
        self.dir_sem2 = os.path.join(base_path, 'labelB_rgb')
        self.dir_bcd = os.path.join(base_path, 'label_bcd')
        
        # Validate directories exist
        for d in [self.dir_A, self.dir_B, self.dir_sem1, self.dir_sem2, self.dir_bcd]:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Directory not found: {d}")

        # 3. Get list of valid filenames
        self.file_list = self._get_valid_file_list()
        print(f"[{dataset_name}] {mode} set loaded. Found {len(self.file_list)} valid samples.")

    def _get_valid_file_list(self):
        # List all files in directory A
        files = [f for f in os.listdir(self.dir_A) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
        valid_files = []
        
        # Ensure file exists in all paired directories
        for f in files:
            if (os.path.exists(os.path.join(self.dir_B, f)) and
                os.path.exists(os.path.join(self.dir_sem1, f)) and 
                os.path.exists(os.path.join(self.dir_sem2, f)) and
                os.path.exists(os.path.join(self.dir_bcd, f))):
                valid_files.append(f)
                
        return valid_files

    def _sync_transform(self, img_A, img_B, sem1, sem2, bcd):
        # Random Horizontal Flip
        if random.random() > 0.5:
            img_A = TF.hflip(img_A)
            img_B = TF.hflip(img_B)
            sem1 = TF.hflip(sem1)
            sem2 = TF.hflip(sem2)
            bcd = TF.hflip(bcd)

        # Random Vertical Flip
        if random.random() > 0.5:
            img_A = TF.vflip(img_A)
            img_B = TF.vflip(img_B)
            sem1 = TF.vflip(sem1)
            sem2 = TF.vflip(sem2)
            bcd = TF.vflip(bcd)
            
        # Random Rotation (0, 90, 180, 270)
        rotations = [0, 90, 180, 270]
        angle = random.choice(rotations)
        if angle > 0:
            img_A = TF.rotate(img_A, angle)
            img_B = TF.rotate(img_B, angle)
            sem1 = TF.rotate(sem1, angle)
            sem2 = TF.rotate(sem2, angle)
            bcd = TF.rotate(bcd, angle)
            
        return img_A, img_B, sem1, sem2, bcd

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # 1. Load Images
        path_A = os.path.join(self.dir_A, filename)
        path_B = os.path.join(self.dir_B, filename)
        img_A = io.imread(path_A)
        img_B = io.imread(path_B)

        # 2. Load Labels
        path_sem1 = os.path.join(self.dir_sem1, filename)
        path_sem2 = os.path.join(self.dir_sem2, filename)
        path_bcd = os.path.join(self.dir_bcd, filename)
        
        label_sem1 = io.imread(path_sem1)
        label_sem2 = io.imread(path_sem2)
        label_bcd = io.imread(path_bcd)

        # 3. Process Semantic Labels (RGB -> Index)
        label_sem1 = Color2Index(label_sem1, self.lookup_table, self.num_classes)
        label_sem2 = Color2Index(label_sem2, self.lookup_table, self.num_classes)

        # 4. Process BCD Label
        if len(label_bcd.shape) == 3:
            label_bcd = label_bcd[:, :, 0]
        label_bcd = (label_bcd > 0).astype(np.int64)

        # 5. Normalize Images
        img_A = norm_utils.normalize_image(img_A, 'A', self.dataset_name)
        img_B = norm_utils.normalize_image(img_B, 'B', self.dataset_name)

        # 6. Convert to Tensors
        # [FIX] Ensure images are float32 to avoid double/float mismatch with model weights
        t_img_A = TF.to_tensor(img_A).float()
        t_img_B = TF.to_tensor(img_B).float()
        
        # Labels need to be (1, H, W) for spatial transforms to work
        t_sem1 = torch.from_numpy(label_sem1).long().unsqueeze(0)
        t_sem2 = torch.from_numpy(label_sem2).long().unsqueeze(0)
        t_bcd = torch.from_numpy(label_bcd).float().unsqueeze(0)

        # 7. Apply Augmentations (only for training)
        if self.mode == 'train' and self.random_flip:
            t_img_A, t_img_B, t_sem1, t_sem2, t_bcd = self._sync_transform(
                t_img_A, t_img_B, t_sem1, t_sem2, t_bcd
            )
            
        if self.mode == 'train' and self.random_swap:
            if random.random() > 0.5:
                t_img_A, t_img_B = t_img_B, t_img_A
                t_sem1, t_sem2 = t_sem2, t_sem1

        # Squeeze back to (H, W) for CrossEntropy / Loss functions
        t_sem1 = t_sem1.squeeze(0)
        t_sem2 = t_sem2.squeeze(0)
        t_bcd = t_bcd.squeeze(0)

        return {
            'img_A': t_img_A,
            'img_B': t_img_B,
            'sem1': t_sem1,
            'sem2': t_sem2,
            'bcd': t_bcd,
            'filename': filename
        }

    def __len__(self):
        return len(self.file_list)