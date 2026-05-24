import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Read data

data_dir = r"D:/medical_paper/Inpaint-Anything/segment-anything-2/data/"  # Path to dataset (LabPics 1)
data = []  # list of files in dataset
for ff, name in enumerate(os.listdir(data_dir + "image/")):  # go over all folder annotation
    data.append({"image": data_dir + "image/" + name,
                 "annotation": data_dir + "mask2/" + name})


# 加载训练批次的主要函数
# 训练批次包括：一张图像（Img）、对应图像中分割的二进制掩码列表（masks），以及每个掩码内单个点的坐标（points）
def read_batch(data):  # read random image and its annotaion from  the dataset (LabPics)

    #  select image

    ent = data[np.random.randint(len(data))]  # choose random entry
    Img = cv2.imread(ent["image"])[..., ::-1]  # read image
    ann_map = cv2.imread(ent["annotation"])  # read annotation

    if ann_map is None:
        print("Failed to load image")

    # resize image

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                         interpolation=cv2.INTER_NEAREST)

    # merge vessels and materials annotations

    mat_map = ann_map[:, :, 0]  # material annotation map
    ves_map = ann_map[:, :, 2]  # vessel  annotaion map
    mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)  # merge maps

    # Get binary masks and points

    inds = np.unique(mat_map)[1:]  # load all indices
    points = []
    masks = []
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # make binary mask corresponding to index ind
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))])  # choose random point/coordinate
        points.append([[yx[1], yx[0]]])
    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])


# Load model 加载SAM模型

checkpoint = "D:/medical_paper/Inpaint-Anything/segment-anything-2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device="cpu")  # load model
predictor = SAM2ImagePredictor(sam2_model)

# 只训练掩码解码器和提示编码器
# 提示编码器处理输入点
# 掩码解码器接收图像编码器和提示编码器的输出，并生成最终的分割掩码
# Set training parameters

predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler()  # mixed precision

# Training loop
# 10为训练轮数
for itr in range(250000):
    with torch.cuda.amp.autocast():  # cast to mix precision
        image, mask, input_point, input_label = read_batch(data)  # load data batch
        if mask.shape[0] == 0:
            continue  # ignore empty batches
        predictor.set_image(image)  # apply SAM image encoder to the image 加载图像并传递给图像编码器

        # prompt encoding

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None,
                                                                                mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),
                                                                                 boxes=None, masks=None, )

        # mask decoder

        batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

        # 运行网络的mask_decoder部分并生成3个分割掩码(low_res_masks)及其分数(prd_scores)
        # prd_masks 包含每个输入点的3个预测掩码,但我们只会使用每个点的第一个掩码
        # prd_scores 包含网络认为每个掩码有多好(或对预测有多确定)的分数

        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, multimask_output=True, repeat_image=batched_mode,
            high_res_features=high_res_features, )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[
            -1])  # Upscale the masks to the original image resolution

        # Segmentation Loss calculation
        device = 'cpu'
        gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
        prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map 将真实掩码转换为torch张量
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log(
            (1 - prd_mask) + 0.00001)).mean()  # cross entropy loss

        # Score loss 分数损失 calculation (intersection over union) IOU

        # 通过使用交集除并集(IOU)指标比较GT掩码和相应的预测掩码，得到预测掩码的真实分数
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        # IOU简单来说就是两个掩码的重叠区域除以两个掩码的合并区域
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        # 使用IOU作为每个掩码的真实分数,并将分数损失作为预测分数与我们刚刚计算的IOU之间的绝对差异
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        # 最后合并分割损失和分数损失(给予前者更高的权重)
        loss = seg_loss + score_loss * 0.05  # mix losses

        # apply back propogation 反向传播

        predictor.model.zero_grad()  # empty gradient
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update()  # Mix precision

        # 每1000步保存一次训练好的模型
        if itr % 1000 == 0:
            torch.save(predictor.model.state_dict(), "model.torch")
            print("save model")

        # Display results
        # 由于我们已经计算了IOU,可以将其显示为移动平均值,查看模型预测随时间的改善情况
        # 大约25,000步后,应该会看到重大改进！

        if itr == 0:
            mean_iou = 0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        print("step)", itr, "Accuracy(IOU)=", mean_iou)
