import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import gradio as gr
import os
import math

# 1. SETUP THIẾT BỊ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. ĐỊNH NGHĨA MODEL (Giữ nguyên kiến trúc chuẩn padding=1)
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

# 3. LOAD MODEL
model = MyCNN().to(device)
model_path = 'best_mnist_model.pth'

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Đã load Model thành công!")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
else:
    print(f"⚠️ CẢNH BÁO: Không tìm thấy file {model_path}!")
model.eval()

# --- HÀM PHỤ TRỢ: TÍNH TRỌNG TÂM (CENTER OF MASS) ---
def get_center_of_mass(img_np):
    # img_np là ảnh grayscale (0-255)
    h, w = img_np.shape
    total_mass = np.sum(img_np)
    if total_mass == 0: return h//2, w//2
    
    # Tạo lưới tọa độ
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Tính trọng tâm
    cy = np.sum(y_coords * img_np) / total_mass
    cx = np.sum(x_coords * img_np) / total_mass
    return cy, cx

# 4. HÀM XỬ LÝ ẢNH (PHIÊN BẢN CENTER OF MASS)
def smart_preprocess(input_data):
    if input_data is None: return None, None
    
    # A. Lấy ảnh
    if isinstance(input_data, dict):
        img = input_data.get('composite')
    else:
        img = input_data
        
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))

    # B. Xử lý nền
    if img.mode == 'RGBA':
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    
    img = img.convert('L')
    img = ImageOps.invert(img) # Nền đen, chữ trắng
    
    # C. Cắt vùng thừa (Bounding Box)
    bbox = img.getbbox()
    if bbox is None: return None, None
    img_cropped = img.crop(bbox)
    
    # D. Resize về 20x20 (Giữ tỷ lệ)
    width, height = img_cropped.size
    if width > height:
        new_height = int(20 * height / width)
        img_resized = img_cropped.resize((20, new_height), Image.Resampling.LANCZOS)
    else:
        new_width = int(20 * width / height)
        img_resized = img_cropped.resize((new_width, 20), Image.Resampling.LANCZOS)
        
    # E. Tạo canvas 28x28 tạm thời và dán vào giữa (theo hình học)
    temp_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - img_resized.width) // 2
    paste_y = (28 - img_resized.height) // 2
    temp_img.paste(img_resized, (paste_x, paste_y))
    
    # F. DỊCH CHUYỂN THEO TRỌNG TÂM (BƯỚC QUAN TRỌNG NHẤT)
    img_np = np.array(temp_img)
    cy, cx = get_center_of_mass(img_np)
    
    # Tính độ lệch so với tâm ảnh (14, 14)
    shift_y = 14 - cy
    shift_x = 14 - cx
    
    # Dịch chuyển ảnh
    # Dùng Affine Transform của PIL để dịch chuyển mượt mà
    final_img = temp_img.transform(
        temp_img.size, 
        Image.AFFINE, 
        (1, 0, -shift_x, 0, 1, -shift_y), # Ma trận dịch chuyển ngược
        resample=Image.Resampling.BICUBIC
    )

    # G. Chuyển sang Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(final_img).unsqueeze(0).to(device), final_img

# 5. DỰ ĐOÁN
def predict(sketch):
    try:
        tensor, processed_img = smart_preprocess(sketch)
        if tensor is None: return {"Chưa vẽ": 0.0}, None
        
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
        
        result = {str(i): float(probs[i]) for i in range(10)}
        return result, processed_img 

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return {"Lỗi": 0.0}, None

# 6. GIAO DIỆN
if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## ✍️ AI Nhận Diện 1 Chữ Số ")
        
        with gr.Row():
            with gr.Column():
                # Giảm brush size xuống một chút để nét 6 không bị dính thành cục
                inp = gr.Sketchpad(label="Vẽ số vào đây", type="numpy", brush=gr.Brush(default_size=15))
                btn = gr.Button("Dự đoán", variant="primary")
            
            with gr.Column():
                out_label = gr.Label(num_top_classes=3, label="Kết quả")
                out_image = gr.Image(label="Ảnh sau khi căn trọng tâm (Input thực tế)", type="pil", height=200)

        btn.click(fn=predict, inputs=inp, outputs=[out_label, out_image])

    demo.launch()