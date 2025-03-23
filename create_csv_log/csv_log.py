import numpy as np
import pandas as pd
import os

def compute_z(data):
    data = np.array(data)
    mean = np.mean(data, axis=1).round(1)
    x_mean = round(np.sum(mean) / len(mean), 2)
    std = np.std(mean)
    lower = x_mean - 1.96 * std / np.sqrt(len(mean))
    if lower - int(lower) > 0.5:
        return np.ceil(lower)
    elif lower - int(lower) < 0.5:
        return np.floor(lower)
    else:
        return lower

# Đường dẫn thư mục gốc và file CSV chứa dữ liệu
root_dir = 'fish_label'
csv_file = 'SNI_file/SNI.csv'
output_csv = 'data/fish_scores.csv'  # File CSV để lưu kết quả

# Đọc dữ liệu từ file CSV
df = pd.read_csv(csv_file)
df.drop(index=[0, 1], inplace=True)

# Đặt tên các cột liên quan đến từng loại cá
mackerel_cols = ["Day", "Sess.", "Mackerel"] + [f"Mackerel.{i}" for i in range(1, 6)]
tilapia_cols = ["Day", "Sess.", "Tilapia"] + [f"Tilapia.{i}" for i in range(1, 6)]
tuna_cols = ["Day", "Sess.", "Tuna"] + [f"Tuna.{i}" for i in range(1, 6)]

# Nhóm dữ liệu theo ngày
group = df.groupby("Day")

# Danh sách lưu kết quả
results = []

for i in range(1, 12):
    fish = pd.DataFrame(group.get_group(f"{i}"), dtype=np.float64).reset_index(drop=True)
    mackerel = fish[mackerel_cols]
    tilapia = fish[tilapia_cols]
    tuna = fish[tuna_cols]

    print(f"Processing Day {i}...")
    day_folder = f"Day {i}"

    for j in range(0, 4, 2):
        res_mackerel = compute_z(mackerel.iloc[j:j+2, 2:])
        res_tilapia = compute_z(tilapia.iloc[j:j+2, 2:])
        res_tuna = compute_z(tuna.iloc[j:j+2, 2:])

        session_index = (j // 2) + 1
        session_folder = f"Session {session_index}"

        fish_data = [("Mackerel", res_mackerel), 
                     ("Tilapia", res_tilapia), 
                     ("Tuna", res_tuna)]

        for fish_name, score in fish_data:
            fish_path = os.path.join(root_dir, day_folder, session_folder, fish_name)
            if os.path.exists(fish_path):
                for img_name in os.listdir(fish_path):
                    if img_name.endswith((".jpg", ".png", ".jpeg")):  # Chỉ lấy file ảnh
                        results.append([img_name, score])
            else:
                print(f"Folder not found: {fish_path}")

# Lưu kết quả vào file CSV
df_results = pd.DataFrame(results, columns=["filename", "score"])
df_results.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")
