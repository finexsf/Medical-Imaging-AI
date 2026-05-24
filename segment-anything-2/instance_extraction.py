import os
import glob
import shutil


def rename_and_save_bmode_images(root_dir, image_names, save_dir):
    # 检查保存目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化子文件夹计数器
    folder_count = 0

    # 遍历根目录中的子目录
    for sub_dir in glob.glob(os.path.join(root_dir, '*')):
        if os.path.isdir(sub_dir):  # 确保遍历的是目录
            folder_count += 1  # 计数器加1
            for img_name in image_names:
                img_paths = glob.glob(os.path.join(sub_dir, img_name))
                for img_path in img_paths:
                    # 获取路径中最后一个部分（通常是文件夹名称）作为文件名
                    renamed_img_name = os.path.basename(sub_dir)

                    # 保存路径，并将文件名更改为提取的名称
                    save_path = os.path.join(save_dir, f"{renamed_img_name}.jpg")

                    # 复制并重命名图像
                    shutil.copy(img_path, save_path)
                    print(f"Saved image: {save_path}")
    print(f"Processed {folder_count} subfolders.")


root_dir = r"D:\medical_paper\Inpaint-Anything\results\2012(2)\2012\ALN2"  # 病历号的上一级
image_names = ["mask_0.jpg"]  # 注意格式：jpg/png！！！201007240033(第一张图片)是jpg格式，其余都是png格式！
save_dir = r"D:\medical_paper\Inpaint-Anything\data\mask"  # 保存的文件夹

rename_and_save_bmode_images(root_dir, image_names, save_dir)
