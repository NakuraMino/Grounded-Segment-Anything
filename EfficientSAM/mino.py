import zipfile
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

models = {}


# Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
with zipfile.ZipFile("./efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")

# # Build the EfficientSAM-Ti model.
# models['efficientsam_ti'] = build_efficient_sam_vitt()

# Build the EfficientSAM-S model.
models['efficientsam_s'] = build_efficient_sam_vits()