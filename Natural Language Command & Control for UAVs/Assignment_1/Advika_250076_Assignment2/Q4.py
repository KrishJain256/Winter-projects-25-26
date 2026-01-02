import torch

def process_image_mask(images):
    
    processed_tensor = torch.where(images < 0.5, 0.0, 1.0)
    return processed_tensor

images = torch.rand(4, 28, 28)
masked_images = process_image_mask(images)
print(f"Masked Tensor Shape: {masked_images.shape}")
print(f"Unique values after mask: {torch.unique(masked_images)}")