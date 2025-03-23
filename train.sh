#!/bin/bash

# Kích hoạt môi trường ảo (nếu có)
# source venv/bin/activate  # Nếu bạn dùng virtualenv
# conda activate your_env_name  # Nếu bạn dùng Conda

# Chạy script huấn luyện
python train.py --epochs 50 --batch_size 32 --lr 0.001 --gpu

# Lưu mô hình sau khi train
echo "Training completed. Model saved in models/"
