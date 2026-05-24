import os
import glob
import shutil


def rename_and_save_bmode_images(root_dir, subfolders, image_names, save_dir):
    # 检查保存目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        print("save_dir not exist")

    # 遍历根目录中的子目录
    for sub_dir in glob.glob(os.path.join(root_dir, '*')):
        # 获取当前子目录的名称作为文件夹名
        renamed_img_name = os.path.basename(sub_dir)

        # 遍历子文件夹
        for subfolder in subfolders:
            img_dir = os.path.join(sub_dir, subfolder)
            if os.path.isdir(img_dir):
                # 遍历要查找的图像文件名
                for img_name in image_names:
                    img_paths = glob.glob(os.path.join(img_dir, img_name))
                    for img_path in img_paths:
                        # 设置保存路径
                        save_path = os.path.join(save_dir, renamed_img_name)

                        # 复制并重命名图像
                        shutil.copy(img_path, save_path)
                        print(f"Saved image: {save_path}")


# 使用示例
root_dir = r"D:\medical_paper\Inpaint-Anything\2011-1"  # 病历号的上一级
subfolders = ["breast"]  # 病历号的下一级(比如ALN1、ALN2、breast文件夹)
image_names = ["breast1.jpg"]  # 需要提取的图像名字(比如bmode1、bmode1、breast1图片)
save_dir = r"D:\medical_paper\Inpaint-Anything\results\2011-1\breast"  # 保存的文件夹
rename_and_save_bmode_images(root_dir, subfolders, image_names, save_dir)
