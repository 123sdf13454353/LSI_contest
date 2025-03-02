from config import load_model
from model import VAE
import matplotlib.pyplot as plt
import numpy as np
from dataset import load_dataset

vae_encoder_fixedpoint_quantized = "vae_encoder_fixedpoint.h5"
vae_decoder_fixedpoint_quantized = "vae_decoder_fixedpoint.h5"

model_vae_encoder_fixedpoint= load_model(vae_encoder_fixedpoint_quantized)
model_vae_decoder_fixedpoint= load_model(vae_decoder_fixedpoint_quantized)
vae_fixedpoint=VAE(model_vae_encoder_fixedpoint,model_vae_decoder_fixedpoint,beta=0.001)
vae_fixedpoint.compile(optimizer="adam",loss="mse",metrics=['accuracy'])
# Xác định số lượng ảnh
x_train,x_test=load_dataset
num_images = x_print.shape[0]
num_cols = 4  # Số cột trong grid
num_rows = (num_images + num_cols - 1) // num_cols  # Tính số hàng cần thiết

# Tạo figure với kích thước phù hợp
fig, axes = plt.subplots(2 * num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

for i in range(num_images):
    row = i // num_cols  # Xác định hàng trong grid
    col = i % num_cols   # Xác định cột trong grid

    # Lấy ảnh đã xử lý từ x_print
    image_array = x_print[i]  # (64, 64, 3)
    image_array = np.expand_dims(image_array, axis=0)  # Định dạng phù hợp cho model

    # Dự đoán ảnh sử dụng mô hình đã load
    reconstructed_image = vae_fixedpoint.predict(image_array)

    # Tính toán bản đồ lỗi
    error_image = (image_array - reconstructed_image) ** 2
    error_image = np.mean(error_image, axis=-1)

    # # Loại bỏ giá trị nhỏ hơn threshold (giữ lại điểm bất thường)
    threshold = 0.035  # Hoặc dùng Gaussian threshold nếu cần
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