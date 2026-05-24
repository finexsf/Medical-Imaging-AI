import numpy as np
import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
from omegaconf import OmegaConf
from segment_anything_2.sam2.build_sam import build_sam2
from segment_anything_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from lama.saicinpainting.training.trainers import load_checkpoint
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


# 自定义医学数据集类
class MedicalDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        for name in os.listdir(data_dir + "image/"):
            self.data.append({"image": data_dir + "image/" + name,
                              "annotation": data_dir + "mask/" + name})

        # 计算所有数据中最大掩码数量，确保数据集中的掩码数量一致
        self.max_masks = 0
        for item in self.data:
            ann_map = cv2.imread(item["annotation"])
            if ann_map is not None:
                mat_map = ann_map[:, :, 0]
                ves_map = ann_map[:, :, 2]
                mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)
                num_masks = len(np.unique(mat_map)[1:])  # 计算掩码数量
                self.max_masks = max(self.max_masks, num_masks)  # 更新最大掩码数量

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        img_path = ent["image"]
        ann_path = ent["annotation"]

        # 读取图像并将颜色从BGR转换为RGB
        Img = cv2.imread(img_path)[..., ::-1]
        ann_map = cv2.imread(ann_path)

        if ann_map is None:
            print(f"加载图像失败: {img_path}")
            return None, None, None, None

        # 确保图像和注释图都调整为相同的大小
        target_size = (1024, 1024)
        Img = cv2.resize(Img, target_size)
        ann_map = cv2.resize(ann_map, target_size, interpolation=cv2.INTER_NEAREST)

        # 提取掩码
        mat_map = ann_map[:, :, 0]
        ves_map = ann_map[:, :, 2]
        mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)

        inds = np.unique(mat_map)[1:]  # 获取唯一的掩码索引
        points = []
        masks = []
        for ind in inds:
            mask = (mat_map == ind).astype(np.uint8)  # 生成二值掩码
            masks.append(mask)
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:  # 确保存在可用的坐标
                yx = np.array(coords[np.random.randint(len(coords))])  # 选择掩码中的随机点
                points.append([[yx[1], yx[0]]])
            else:
                print(f"没有找到掩码索引 {ind} 的坐标，图像路径: {img_path}")

        # 扩充掩码和点的数量到最大掩码数
        while len(masks) < self.max_masks:
            masks.append(np.zeros(target_size, dtype=np.uint8))  # 添加全零的掩码
            points.append([[-1, -1]])  # 添加无效点

        # 打印图像和掩码的形状以及路径
        # print(f"Image path: {img_path}, Annotation path: {ann_path}")
        # print(f"Image shape: {Img.shape}, Number of masks: {len(masks)}, Masks shape: {[mask.shape for mask in masks]}")

        return Img, np.array(masks).astype(np.float32), np.array(points), np.ones([len(masks), 1])


# 数据集路径和加载
data_dir = r"D:/medical_paper/Inpaint-Anything/data/"
batch_size = 8
dataset = MedicalDataset(data_dir)

# 按 7:1:2 比例划分数据集
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载SAM模型
checkpoint = "D:/medical_paper/Inpaint-Anything/segment_anything_2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(config_file=model_cfg, packpt_th=checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# 设置训练参数
# 将模型的前面部分参数冻结
for param in predictor.model.parameters():
    param.requires_grad = False

# 解冻最后五层
# 这里假设模型的层是按顺序排列的，您需要根据实际情况调整
last_five_layers = list(predictor.model.children())[-5:]  # 获取最后五层
for layer in last_five_layers:
    for param in layer.parameters():
        param.requires_grad = True
predictor.model.sam_mask_decoder.train(True)  # 视情况决定是否将此设置为False
predictor.model.sam_prompt_encoder.train(True)
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler()


# LaMa模型加载函数
def build_lama_model(config_p: str, ckpt_p: str, device="cuda"):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'tensorboard'

    checkpoint_path = os.path.join(predict_config.model.path, 'models',
                                   predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=torch.device('cuda'))
    model.to(device)
    model.freeze()

    # 冻结LaMa模型所有参数
    for param in model.parameters():
        param.requires_grad = False

    return model


# 加载LaMa模型
lama_config = "D:/medical_paper/Inpaint-Anything/lama/configs/prediction/default.yaml"
lama_ckpt = "D:/medical_paper/Inpaint-Anything/lama/checkpoint/big-lama"
lama_model = build_lama_model(lama_config, lama_ckpt, device=device)


def train_one_epoch(epoch, predictor, train_loader, optimizer, scaler):
    predictor.model.train()  # 设置模型为训练模式
    total_loss = 0
    total_iou = 0  # 用于累加 IoU 分数
    num_batches = 0  # 用于计算平均 IoU

    # 使用 tqdm 创建进度条
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

    for batch_idx, (images, masks, input_points, input_labels) in progress_bar:
        images = [image.to(device) for image in images]
        masks = [mask.to(device) for mask in masks]
        input_points = [input_point.to(device) for input_point in input_points]
        input_labels = [input_label.to(device) for input_label in input_labels]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            for i in range(len(images)):
                # 使用SAM模型进行预测
                if isinstance(images[i], torch.Tensor):
                    image_np = images[i].cpu().numpy()  # 转换为numpy数组
                    predictor.set_image(image_np)

                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_points[i], input_labels[i], box=None, mask_logits=None, normalize_coords=True
                )

                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )
                batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features
                )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                # SAM生成的预测掩码
                gt_mask = masks[i]
                prd_mask = torch.sigmoid(prd_masks[:, 0])

                # 计算损失
                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()

                inter = (gt_mask * (prd_mask > 0.5)).sum()
                iou = inter / (gt_mask.sum() + (prd_mask > 0.5).sum() - inter + 1e-5)  # 添加一个小的常数以避免除以零
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                loss = seg_loss + score_loss * 0.05
                total_loss += loss.item()
                total_iou += iou.item()  # 累加 IoU 分数
                num_batches += 1  # 计数当前 batch

                # 反向传播与优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # 更新进度条的描述
        avg_iou = total_iou / num_batches if num_batches > 0 else 0  # 计算平均 IoU
        progress_bar.set_postfix(loss=loss.item(), avg_iou=avg_iou)

        # 打印损失和 IoU，使用 \r 来覆盖
        print(f'\nEpoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg IoU: {avg_iou:.4f}', end='\n')

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / num_batches if num_batches > 0 else 0  # 计算整个 epoch 的平均 IoU
    print(f'\nEpoch {epoch + 1} finished, Average Loss: {avg_loss:.4f}, Average IoU: {avg_iou:.4f}')


# 验证函数
def validate(predictor, val_loader, best_iou):
    predictor.model.eval()  # 设置模型为验证模式
    total_loss = 0
    total_iou = 0
    num_batches = 0
    best_iou_score = best_iou  # 当前最佳 IoU

    with torch.no_grad():
        # 使用 tqdm 创建进度条
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validating')

        for batch_idx, (images, masks, input_points, input_labels) in progress_bar:
            images = [image.to(device) for image in images]
            masks = [mask.to(device) for mask in masks]
            input_points = [input_point.to(device) for input_point in input_points]
            input_labels = [input_label.to(device) for input_label in input_labels]

            for i in range(len(images)):
                if isinstance(images[i], torch.Tensor):
                    image_np = images[i].cpu().numpy()  # 转换为numpy数组
                    predictor.set_image(image_np)

                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_points[i], input_labels[i], box=None, mask_logits=None, normalize_coords=True
                )

                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction

                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features
                )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                gt_mask = masks[i]
                prd_mask = torch.sigmoid(prd_masks[:, 0])

                # 计算损失
                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
                total_loss += seg_loss.item()

                # 计算 IoU
                intersection = torch.sum(gt_mask * prd_mask)
                union = torch.sum(gt_mask) + torch.sum(prd_mask) - intersection
                iou = intersection / (union + 1e-6)  # 避免除以零
                total_iou += iou.item()
                num_batches += 1

            # 更新进度条的描述
            avg_loss = total_loss / (batch_idx + 1)
            avg_iou = total_iou / (num_batches)
            progress_bar.set_postfix(loss=avg_loss, iou=avg_iou)

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    print(f'验证损失: {avg_loss}, 平均 IoU: {avg_iou}')

    # 保存最佳 IoU
    if avg_iou > best_iou_score:
        best_iou_score = avg_iou
        torch.save(predictor.model.state_dict(), f'./pt/best_model_{best_iou_score:.4f}.pth')  # 保存模型
        print(f'保存新最佳模型，IoU: {best_iou_score}')

    return best_iou_score


# 测试函数
def test(predictor, test_loader):
    predictor.model.eval()  # 设置模型为测试模式
    with torch.no_grad():
        for images, masks, input_points, input_labels in test_loader:
            images = [image.to(device) for image in images]
            masks = [mask.to(device) for mask in masks]
            input_points = [input_point.to(device) for input_point in input_points]
            input_labels = [input_label.to(device) for input_label in input_labels]

            for i in range(len(images)):
                predictor.set_image(images[i])
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_points[i],
                                                                                        input_labels[i],
                                                                                        box=None, mask_logits=None, normalize_coords=True)

                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),
                                                                                         boxes=None, masks=None,)
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                     predictor._features["high_res_feats"]]
                batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction

                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True, repeat_image=batched_mode, high_res_features=high_res_features
                )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                prd_mask = torch.sigmoid(prd_masks[:, 0])

                # 可视化或保存结果
                result_image = images[i].cpu().numpy().astype(np.uint8)
                result_mask = prd_mask.cpu().numpy()
                cv2.imshow("Result Image", result_image)
                cv2.imshow("Predicted Mask", result_mask)
                cv2.waitKey(0)


# 训练过程
num_epochs = 1  # 训练轮数
best_iou = 0.0   # 初始化最佳 IoU

for epoch in range(num_epochs):
    train_one_epoch(epoch, predictor, train_loader, optimizer, scaler)
    best_iou = validate(predictor, val_loader, best_iou)

# 测试过程
test(predictor, test_loader)
