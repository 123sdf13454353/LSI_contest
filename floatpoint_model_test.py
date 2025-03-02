from fixedpoint_update import convert_weights_to_fxp,save_weights_to_dict,extract_weights_recursive
from dataset import load_dataset
from config import encoder_floatpoint,decoder_floatpoint,load_model


from model import VAE
import os
import random
import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from PIL import Image

# Đường dẫn tới mô hình đã lưu
# encoder_path = "vae_encoder_fixed_16_8.h5"
# decoder_path = "vae_decoder_fixed_16_8.h5"
# encoder = load_model(ENCODER_PATH)
# decoder = load_model(DECODER_PATH)
# # Lấy danh sách tất cả ảnh trong thư mục

x_train_new, x_test,x_good,x_cut,x_train,x_hole,x_print=load_dataset()
vae_afterload = VAE( encoder_floatpoint, decoder_floatpoint, beta=0.001)
vae_afterload.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

loss, accuracy = vae_afterload.evaluate(x_test, x_test, verbose=1)

print(f"🔍 Đánh giá mô hình:")
print(f"✅ Loss: {loss:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
# Lấy danh sách tất cả ảnh trong thư mục


image_paths = glob.glob('hazelnut/test/print/*.png')

# Xác định số lượng ảnh
num_images = len(image_paths)
num_cols = 4  # Số cột trong grid
num_rows = (num_images + num_cols - 1) // num_cols  # Tính số hàng cần thiết

# Tạo figure với kích thước phù hợp
fig, axes = plt.subplots(2 * num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

for i, image_path in enumerate(image_paths):
    row = i // num_cols  # Xác định hàng trong grid
    col = i % num_cols   # Xác định cột trong grid

    # Đọc ảnh và tiền xử lý
    image = Image.open(image_path).convert("RGB")
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Dự đoán ảnh sử dụng mô hình đã load
    reconstructed_image = vae_afterload.predict(image_array)

    # Tính toán bản đồ lỗi
    error_image = (image_array - reconstructed_image) ** 2
    error_image = np.mean(error_image, axis=-1)

    # # Loại bỏ giá trị nhỏ hơn threshold (giữ lại điểm bất thường)
    threshold = 0.035 # Hoặc dùng Gaussian threshold nếu cần
    error_image[error_image < threshold] = 0

    # Hiển thị ảnh gốc
    axes[2 * row, col].imshow(image_array[0])
    axes[2 * row, col].axis("off")
    axes[2 * row, col].set_title(f"Image {i+1}", fontsize=10)

    # Hiển thị bản đồ lỗi
    axes[2 * row + 1, col].imshow(error_image[0], cmap="gray")
    axes[2 * row + 1, col].axis("off")
    axes[2 * row + 1, col].set_title(f"Anomaly Map {i+1}", fontsize=10)

# Xóa các ô trống nếu tổng số ảnh không chia hết cho số cột
for j in range(num_images, num_cols * num_rows):
    axes[2 * (j // num_cols), j % num_cols].axis("off")
    axes[2 * (j // num_cols) + 1, j % num_cols].axis("off")

plt.tight_layout()
plt.show()
