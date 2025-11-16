import torch

checkpoint_path = '/home/goi-tom/ANT_DC12_UniTrento/models/final_model.pth'

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract state_dict
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

print(f"{'Parameter Name':60} {'Shape':25} {'# Elements':15}")
print("="*100)

# Loop through all parameters
total_params = 0
for name, param in state_dict.items():
    shape = tuple(param.shape) if hasattr(param, 'shape') else (len(param),)
    num_elements = param.numel() if hasattr(param, 'numel') else len(param)
    total_params += num_elements
    print(f"{name:60} {str(shape):25} {num_elements:15}")

print("="*100)
print(f"Total number of parameters: {total_params}")
