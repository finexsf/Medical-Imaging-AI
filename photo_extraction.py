import os
import glob
import shutil


def rename_and_save_bmode_images(root_dir, subfolders, image_names, save_dir):
    # 检查保存目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历根目录中的子目录
    for sub_dir in glob.glob(os.path.join(root_dir, '*')):
        for subfolder in subfolders:
            img_dir = os.path.join(sub_dir, subfolder)

            if os.path.isdir(img_dir):
                for img_name in image_names:
                    img_paths = glob.glob(os.path.join(img_dir, img_name))
                    for img_path in img_paths:
                        # 获取路径中最后一个部分（通常是文件夹名称）作为文件名
                        renamed_img_name = os.path.basename(sub_dir)

                        # 保存路径，并将文件名更改为提取的名称
                        save_path = os.path.join(save_dir, f"{renamed_img_name}.jpg")

                        # 复制并重命名图像
                        shutil.copy(img_path, save_path)
                        print(f"Saved image: {save_path}")


root_dir = r"D:\medical_paper\Inpaint-Anything\2012(2)"  # 病历号的上一级
subfolders = ["ALN2"]  # 病历号的下一级(比如ALN1、ALN2、breast文件夹)
image_names = ["bmode1.jpg"]  # 需要提出的图像名字(比如bmode1、bmode1、breast1图片)
save_dir = r"D:\medical_paper\Inpaint-Anything\example\2012(2)\ALN2"  # 保存的文件夹

rename_and_save_bmode_images(root_dir, subfolders, image_names, save_dir)
