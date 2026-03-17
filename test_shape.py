import torch
import sys

# Simulate modeling_pi05.py scenario
bsize = 1
num_regions = 256
region_patch_size = 1
patches_per_side = 16

def expand_region_mask(region_mask: torch.Tensor, patches_per_side: int, region_patch_size: int) -> torch.Tensor:
    if patches_per_side % region_patch_size != 0:
        raise ValueError(
            f"region_patch_size ({region_patch_size}) must divide patches_per_side ({patches_per_side})."
        )
    batch_size, num_regions_in = region_mask.shape
    region_h = patches_per_side // region_patch_size
    region_w = patches_per_side // region_patch_size
    if num_regions_in != region_h * region_w:
        raise ValueError("Region mask does not match inferred grid.")

    reshaped = region_mask.view(batch_size, region_h, 1, region_w, 1)
    expanded = reshaped.expand(-1, -1, region_patch_size, -1, region_patch_size).contiguous()
    return expanded.view(batch_size, patches_per_side * patches_per_side)

valid_mask = torch.ones(bsize, num_regions, dtype=torch.bool)
global_active_region_mask = valid_mask.clone()

print(f"global_active_region_mask.shape: {global_active_region_mask.shape}")
global_token_mask = expand_region_mask(global_active_region_mask, patches_per_side, region_patch_size)
print(f"global_token_mask.shape: {global_token_mask.shape}")

if global_token_mask.dim() == 1:
    global_token_mask = global_token_mask.unsqueeze(0)
print(f"global_token_mask.shape after unsqueeze: {global_token_mask.shape}")

im = torch.ones(bsize, num_regions, dtype=torch.bool)
m = torch.ones_like(im)
print(f"m.shape: {m.shape}")

try:
    m &= global_token_mask
    print("Success")
except Exception as e:
    print(f"Error: {e}")
