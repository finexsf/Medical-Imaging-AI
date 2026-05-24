import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, original_dir, processed_dir, transform=None):
        self.original_dir = original_dir
        self.processed_dir = processed_dir
        self.transform = transform
        self.original_images = os.listdir(original_dir)

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        original_image_path = os.path.join(self.original_dir, self.original_images[idx])
        processed_image_path = os.path.join(self.processed_dir, self.original_images[idx])

        original_image = Image.open(original_image_path).convert("RGB")
        processed_image = Image.open(processed_image_path).convert("RGB")

        if self.transform:
            original_image = self.transform(original_image)
            processed_image = self.transform(processed_image)

        return original_image, processed_image


# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
dataset = ImageDataset('D:/medical_paper/Inpaint-Anything/example/2010/ALN1', 'D:/medical_paper/Inpaint-Anything/example/2010/ALN1', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# 定义简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 256)  # Assuming input size is 256x256
        self.fc2 = nn.Linear(256, 3 * 256 * 256)  # Output size is 256x256 RGB

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 3, 256, 256)  # Reshape to (batch_size, 3, 256, 256)
        return x


# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for original_images, processed_images in dataloader:
        optimizer.zero_grad()
        outputs = model(original_images)
        loss = criterion(outputs, processed_images)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'image_inpainting_model.pth')

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load('image_inpainting_model.pth'))
model.eval()


# 批量处理图像
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg')):  # 只处理图像文件
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0)  # 添加批次维度

            with torch.no_grad():
                output = model(image)

            output_image = output.squeeze(0).permute(1, 2, 0)  # 改变形状为(H, W, C)
            output_image = (output_image.numpy() * 255).astype('uint8')  # 转换为uint8格式

            # 保存处理后的图像
            output_filename = os.path.splitext(filename)[0] + '.jpg'  # 保持原名，保存为jpg格式
            output_path = os.path.join(output_folder, output_filename)
            Image.fromarray(output_image).save(output_path)


# 使用示例
input_folder = 'D:/medical_paper/Inpaint-Anything/example/2011-1/ALN1'  # 输入文件夹路径
output_folder = 'D:/medical_paper/Inpaint-Anything/results/2011-1/ALN1'  # 输出文件夹路径
process_images(input_folder, output_folder)
