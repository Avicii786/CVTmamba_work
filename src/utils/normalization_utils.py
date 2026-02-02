import numpy as np

# --- SECOND Dataset Constants ---
# Calculated from the training set of the SECOND dataset
SECOND_MEAN_A = np.array([113.40, 114.08, 116.45])
SECOND_STD_A  = np.array([48.30,  46.27,  48.14])

SECOND_MEAN_B = np.array([111.07, 114.04, 118.18])
SECOND_STD_B  = np.array([49.41,  47.01,  47.94])

# --- LandsatSCD Dataset Constants ---
# Calculated from the training set of the LandsatSCD dataset
LANDSAT_MEAN_A = np.array([141.53, 139.20, 137.73])
LANDSAT_STD_A  = np.array([81.99,  83.31,  83.89])

LANDSAT_MEAN_B = np.array([137.36, 136.50, 135.14])
LANDSAT_STD_B  = np.array([85.97,  86.01,  86.81])

def get_constants(dataset_name, time_step):
    """
    Helper to retrieve the correct constants based on dataset and time step.
    
    Args:
        dataset_name (str): Name of the dataset folder (e.g., 'Second_dataset', 'LandsatSCD_dataset').
        time_step (str): 'A' (im1) or 'B' (im2).
        
    Returns:
        tuple: (mean, std) arrays.
    """
    # Normalize input string to handle variations like "Second_dataset" or "SECOND"
    d_name = dataset_name.lower()
    time = time_step.upper()
    
    if 'second' in d_name:
        if time == 'A': return SECOND_MEAN_A, SECOND_STD_A
        if time == 'B': return SECOND_MEAN_B, SECOND_STD_B
        
    elif 'landsat' in d_name:
        if time == 'A': return LANDSAT_MEAN_A, LANDSAT_STD_A
        if time == 'B': return LANDSAT_MEAN_B, LANDSAT_STD_B
    
    # Fallback to standard ImageNet statistics if dataset unknown
    # Mean: [0.485, 0.456, 0.406] * 255 -> [123.675, 116.28, 103.53]
    # Std:  [0.229, 0.224, 0.225] * 255 -> [58.395, 57.12, 57.375]
    print(f"Warning: Unknown dataset '{dataset_name}'. Using standard ImageNet normalization.")
    imagenet_mean = np.array([123.675, 116.28, 103.53])
    imagenet_std = np.array([58.395, 57.12, 57.375])
    return imagenet_mean, imagenet_std

def normalize_image(image, time_step, dataset_name):
    """
    Normalizes an image array using constants from the specified dataset.
    Formula: (image - mean) / std
    
    Args:
        image (numpy.ndarray): Input image array of shape (H, W, 3).
        time_step (str): 'A' or 'B'.
        dataset_name (str): Dataset name.
        
    Returns:
        numpy.ndarray: Normalized image (float32).
    """
    mean, std = get_constants(dataset_name, time_step)
        
    # Ensure image is float for division
    image = image.astype(np.float32)
    
    # Apply normalization (broadcasting handles shape differences)
    normalized_image = (image - mean) / std
    
    return normalized_image

def denormalize_image(normalized_image, time_step, dataset_name):
    """
    Denormalizes an image array to get back original pixel values.
    Formula: (normalized * std) + mean
    
    Args:
        normalized_image (numpy.ndarray): Normalized image array.
        time_step (str): 'A' or 'B'.
        dataset_name (str): Dataset name.
        
    Returns:
        numpy.ndarray: Denormalized image (uint8, clipped to 0-255).
    """
    mean, std = get_constants(dataset_name, time_step)
        
    image = (normalized_image * std) + mean
    
    # Clip values to valid image range and convert to integer
    return np.clip(image, 0, 255).astype(np.uint8)