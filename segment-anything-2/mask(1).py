import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils import show_anns, show_anns_only, load_image_as_mask, calculate_loss, print_pixel_value_ranges

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )



# Display image
image = Image.open('D:/medical_paper/Inpaint-Anything/data/image/201007270220.jpg')
image = np.array(image.convert("RGB"))
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

ground_truth_path = 'D:/medical_paper/Inpaint-Anything/data/mask/201007270220.jpg'
# Load the ground truth mask
ground_truth = load_image_as_mask(ground_truth_path)
plt.figure(figsize=(10, 10))
plt.imshow(ground_truth, cmap='gray')
plt.axis('on')
plt.show()

checkpoint = "D:/medical_paper/Inpaint-Anything/segment-anything-2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
    masks = mask_generator.generate(image)
    print(len(masks))
    print(masks[0].keys())
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    show_anns_only(masks, black_white=True)  # Show only the masks (black and white)

    # mask_generator_2 = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=128,
    #     points_per_batch=128,
    #     pred_iou_thresh=0.8,
    #     stability_score_thresh=0.95,
    #     stability_score_offset=0.9,
    #     crop_n_layers=2,
    #     box_nms_thresh=0.7,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=25.0,
    #     use_m2m=True,
    # )
    # masks2 = mask_generator_2.generate(image)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # show_anns(masks2)
    # plt.axis('off')
    # plt.show()

    masks2 = mask_generator_2.generate(image)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    plt.show()
    show_anns_only(masks2, black_white=True)  # Show only the masks (black and white)
    loss = calculate_loss(masks2, ground_truth)
    print(f"Loss: {loss}")
