import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FishDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Kiểm tra dữ liệu đầu vào
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Thư mục ảnh '{img_dir}' không tồn tại.")
        if self.data.empty:
            raise ValueError(f"File CSV '{csv_file}' không chứa dữ liệu.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # Tên ảnh
        
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy ảnh '{img_name}'.")

        # Lấy nhãn và chuyển đổi
        label = self.data.iloc[idx, 1]  # Giả sử nhãn nằm ở cột thứ hai
        label = int(label)  # Chuyển nhãn thành số nguyên
        label = label - 2  # Chuyển nhãn từ [2, 9] về [0, 7]
        label = torch.tensor(label, dtype=torch.long)  # Định dạng cho CrossEntropyLoss
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transform cho ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Lật ngang ngẫu nhiên
    transforms.RandomRotation(10),  # Xoay ngẫu nhiên
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
])
