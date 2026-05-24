import numpy as np
import torch
import cv2
import os
import sys
import yaml

from omegaconf import OmegaConf
from segment_anything_2.sam2.build_sam import build_sam2
from segment_anything_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from lama.saicinpainting.training.trainers import load_checkpoint


# 设置线程数限制
# os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Read data

data_dir = r"D:/medical_paper/Inpaint-Anything/data/"
data = []
for ff, name in enumerate(os.listdir(data_dir + "image/")):
    data.append({"image": data_dir + "image/" + name,
                 "annotation": data_dir + "mask/" + name})  # 读取随机图像及其标注


# 加载训练批次的主要函数
# 训练批次包括：图像（Img）、分割掩码（masks），以及每个掩码中的一个随机点坐标（points）

def read_batch(data):
    #  选择图像

    ent = data[np.random.randint(len(data))]  # 选择随机条目
    Img = cv2.imread(ent["image"])[..., ::-1]  # 读取图像
    ann_map = cv2.imread(ent["annotation"])  # 读取标注

    if ann_map is None:
        print("Failed to load image")

    #  调整图像大小

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # 缩放因子
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                         interpolation=cv2.INTER_NEAREST)

    # 合并容器和材料标注

    mat_map = ann_map[:, :, 0]  # 材料标注地图
    ves_map = ann_map[:, :, 2]  # 容器标注地图
    mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)  # 合并地图

    # 获取二进制掩码和点

    inds = np.unique(mat_map)[1:]  # 加载地图中所有索引的列表
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # 为索引制作二进制掩码
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # 获取掩码中的所有坐标
        yx = np.array(coords[np.random.randint(len(coords))])  # 选择随机点/坐标
        points.append([[yx[1], yx[0]]])
    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])


# Load model 加载SAM模型

checkpoint = "D:\medical_paper\Inpaint-Anything\segment_anything_2\checkpoints\sam2_hiera_small.pt"  # 模型权重路径
model_cfg = "sam2_hiera_s.yaml"  # 模型配置
sam2_model = build_sam2(config_file=model_cfg, packpt_th=checkpoint, device="cuda")  # 加载模型
predictor = SAM2ImagePredictor(sam2_model)  # 加载网络
# 获取所有参数及其名称
all_params = list(predictor.named_parameters())
# 获取模型层名称，识别最后5层
layer_names = [name for name, _ in all_params]
last_5_layers = layer_names[-5:]  # 获取最后5层的名称
# 遍历所有参数
for name, param in all_params:
    param.requires_grad = False  # 默认冻结所有参数
    # 解冻最后5层的参数
    if any(layer_name in name for layer_name in last_5_layers):
        param.requires_grad = True

    print(f"{name}: {param.requires_grad}")
# 定义一个fliter，只传入requires_grad=True的模型参数

# 只训练掩码解码器和提示编码器(不训练图像编码器)
# 提示编码器处理输入点
# 掩码解码器接收图像编码器和提示编码器的输出，并生成最终的分割掩码
# Set training parameters

predictor.model.sam_mask_decoder.train(True)  # 启用掩码解码器的训练
predictor.model.sam_prompt_encoder.train(True)  # 启用提示编码器的训练
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)  # 定义adamW优化器
scaler = torch.cuda.amp.GradScaler()  # 使用混合精度训练


def build_lama_model(
        config_p: str,
        ckpt_p: str,
        device="cuda"
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
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=torch.device('cuda'))
    model.to(device)
    model.freeze()
    return model


lama_config = "D:/medical_paper/Inpaint-Anything/lama/configs/prediction/default.yaml"
lama_ckpt = "D:/medical_paper/Inpaint-Anything/lama/checkpoint/big-lama"
lama_model = build_lama_model(lama_config, lama_ckpt, device=device)
for name, param in lama_model.named_parameters():
    param.requires_grad = False
    for k in [31, 30, 34, 'model.norm.bias', 'model.norm.weight']:
        if str(k) in name:
            param.requires_grad = True
    # print(f"{name}: {param.requires_grad}")
# 定义一个fliter，只传入requires_grad=True的模型参数
optimizer = torch.optim.AdamW(filter(lambda param: param.requires_grad, lama_model.parameters()), lr=0.01)

# 第一步：读取当前模型参数
model_dict = lama_model.state_dict()
optimizer.zero_grad()
a = torch.tensor([2.], requires_grad=False)  # requires_grad 设置为 False 表示不需要计算这个张量的梯度
loss = lama_model(a)  # 前向传播计算损失
loss.backward()  # 反向传播计算梯度
optimizer.step()  # 根据梯度更新模型参数
# model_weights = lama_model.state_dict()

# 第二步：读取预训练模型
pretrained_dict = torch.load("model.torch", map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
# 第三步：使用预训练的模型更新当前模型参数
model_dict.update(pretrained_dict)
# 第四步：加载模型参数
model_weights = lama_model.load_state_dict(model_dict)

# 训练主循环

for itr in range(25000):  # 10为训练轮数
    with torch.cuda.amp.autocast():  # 转换为混合精度
        image, mask, input_point, input_label = read_batch(data)  # 加载数据批次
        if mask.shape[0] == 0:
            continue  # 忽略空批次
        predictor.set_image(image)  # 对图像应用SAM图像编码器

        # prompt encoding
        # 使用网络的prompt编码器处理输入点

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None,
                                                                                mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),
                                                                                 boxes=None, masks=None, )

        # mask decoder
        # 已经编码了prompt(点)和图像,可以预测分割掩码了

        batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

        # 这段代码的主要部分是 model.sam_mask_decode，它运行网络的mask_decoder部分并生成3个分割掩码(low_res_masks)及其分数(prd_scores)
        # prd_masks 包含每个输入点的3个预测掩码,但我们只会使用每个点的第一个掩码
        # prd_scores 包含网络认为每个掩码有多好(或对预测有多确定)的分数

        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode,
            high_res_features=high_res_features)
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

        # 分割损失
        # 即预测掩码与真实掩码相比有多好
        gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
        prd_mask = torch.sigmoid(prd_masks[:, 0])  # 使用sigmoid函数将预测掩码(prd_mask)从logits转换为概率，将真实掩码转换为torch张量
        # 使用真实掩码(gt_mask)和预测概率图(prd_mask)计算交叉熵损失(seg_loss)
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log(
            (1 - prd_mask) + 0.00001)).mean()  # cross entropy loss

        # 分数损失
        # IOU: 通过使用交集除并集(IOU)指标比较GT掩码和相应的预测掩码，得到预测掩码的真实分数，即预测掩码实际上有多好

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        # 使用IOU作为每个掩码的真实分数,并将分数损失作为预测分数，计算与刚刚计算的IOU之间的绝对差异
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        # 最后合并分割损失和分数损失(给予前者更高的权重)
        loss = seg_loss + score_loss * 0.05  # 混合损失

        # 反向传播
        # 一旦得到损失就可以使用之前创建的优化器计算反向传播并更新权重！

        predictor.model.zero_grad()  # 清空梯度
        scaler.scale(loss).backward()  # 反向传播
        scaler.step(optimizer)
        scaler.update()  # 混合精度

        # 获取权重字典
        predictor_weights = predictor.model.state_dict()

        # 合并权重字典
        combined_weights = {**model_weights, **predictor_weights}

        # 每1000步保存一次训练好的模型
        if itr % 1000 == 0:
            torch.save(combined_weights, "model.torch")
            print("save model")

        # Display results
        # 由于我们已经计算了IOU,可以将其显示为移动平均值,查看模型预测随时间的改善情况
        # 大约25,000步后,应该会看到重大改进！

        if itr == 0:
            mean_iou = 0
        # mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        mean_iou = mean_iou * 0.99 + 0.01 * iou.detach().cuda().mean()
        print("step)", itr, "Accuracy(IOU)=", mean_iou)
