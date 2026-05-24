import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

# 设置线程数限制
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.data import pad_tensor_to_modulo

from utils import load_img_to_array, save_array_to_img


@torch.no_grad()  # 使用@torch.no_grad()装饰器表示在此函数中不需要计算梯度
def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        ckpt_p: str,
        mod=8,
        device="cpu"
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    # 将NumPy数组转换为PyTorch张量
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    # 将模型设置为仅进行预测，并启用可视化器
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'tensorboard'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


# 构建LAMA模型
def build_lama_model(
        config_p: str,
        ckpt_p: str,
        device="cpu"
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    # 将模型设置为仅进行预测，并启用可视化器
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'tensorboard'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=torch.device('cpu'))
    model.to(device)
    model.freeze()
    return model


# 类似于之前的修复函数，但接收已构建的模型作为参数
@torch.no_grad()
def inpaint_img_with_builded_lama(
        model,
        img: np.ndarray,
        mask: np.ndarray,
        config_p=None,
        mod=8,
        device="cpu"
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch["inpainted"][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


def setup_args(parser):
    parser.add_argument(
        "--input_img_dir", type=str,
        default="D:/medical_paper/Inpaint-Anything/lama/data/image",
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "--input_mask_dir", type=str,
        default="D:/medical_paper/Inpaint-Anything/lama/data/mask",
        help="Path to the directory containing input masks.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="D:/medical_paper/Inpaint-Anything/lama/results",
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="D:/medical_paper/Inpaint-Anything/lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="D:/medical_paper/Inpaint-Anything/lama/checkpoint/big-lama",
        help="The path to the lama checkpoint.",
    )


if __name__ == "__main__":
    """Example usage:
    python lama_inpaint.py \
        --input_img FA_demo/FA1_dog.png \
        --input_mask_glob "results/FA1_dog/mask*.png" \
        --output_dir results \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_dir = Path(args.input_img_dir)
    mask_dir = Path(args.input_mask_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_lama_model(args.lama_config, args.lama_ckpt, device=device)
    torch.save(model, "model.torch")
    print("save model")

    # Iterate over all images in the image directory
    for img_file in img_dir.glob("*.jpg"):  # Adjust the extension if needed
        img_stem = img_file.stem
        img = load_img_to_array(img_file)

        # Find corresponding masks
        mask_files = sorted(mask_dir.glob(f"{img_stem}-ALN1.jpg"))  # 调整文件匹配模式!

        for mask_file in mask_files:
            mask = load_img_to_array(mask_file)
            img_inpainted_p = out_dir / f"{img_stem}-ALN1.jpg"
            img_inpainted = inpaint_img_with_lama(img, mask, args.lama_config, args.lama_ckpt, device=device)
            save_array_to_img(img_inpainted, img_inpainted_p)
