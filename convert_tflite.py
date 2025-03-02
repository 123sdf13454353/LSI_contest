from config import load_model, tqdm, Fxp, np, cv2, os, glob,plt
from dataset import load_dataset
import time
import tensorflow as tf

# Load mô hình Keras
x_train_new, x_test, x_good, x_cut, x_train, x_hole, x_print = load_dataset()

encoder_fixedpoint = tf.keras.models.load_model('vae_encoder_fixedpoint.h5')
decoder_fixedpoint = tf.keras.models.load_model('vae_decoder_fixedpoint.h5')

converter_encoder = tf.lite.TFLiteConverter.from_keras_model(encoder_fixedpoint)
converter_decoder = tf.lite.TFLiteConverter.from_keras_model(decoder_fixedpoint)

# Chuyển đổi mô hình sang TFLite mà không lượng tử hóa
tflite_model_encoder = converter_encoder.convert()
tflite_model_decoder = converter_decoder.convert()

# Lưu model TFLite
with open('model_encoder_float.tflite', 'wb') as f:
    f.write(tflite_model_encoder)

with open('model_decoder_float.tflite', 'wb') as f:
    f.write(tflite_model_decoder)

print("✅ Mô hình đã được chuyển đổi sang .tflite mà không áp dụng lượng tử hóa!")

# Load mô hình TFLite (Encoder và Decoder riêng biệt)
interpreter_encoder = tf.lite.Interpreter(model_path='model_encoder_float.tflite')
interpreter_decoder = tf.lite.Interpreter(model_path='model_decoder_float.tflite')

interpreter_encoder.allocate_tensors()
interpreter_decoder.allocate_tensors()

# Lấy thông tin tensor input và output
encoder_input_details = interpreter_encoder.get_input_details()
encoder_output_details = interpreter_encoder.get_output_details()

decoder_input_details = interpreter_decoder.get_input_details()
decoder_output_details = interpreter_decoder.get_output_details()

# Chuyển input về đúng dtype (FLOAT32)
input_data = np.array([x_print[0]], dtype=np.float32)  # Giữ nguyên shape nhưng đổi dtype

# Đồng bộ hóa CPU trước khi đo thời gian
tf.keras.backend.clear_session()

# 🕒 Bắt đầu đo thời gian inference
start_time = time.perf_counter()

# Đưa dữ liệu vào Encoder
interpreter_encoder.set_tensor(encoder_input_details[0]['index'], input_data)
interpreter_encoder.invoke()

# Lấy output từ Encoder (latent vector)
latent_vector = interpreter_encoder.get_tensor(encoder_output_details[0]['index'])

# Đưa latent vector vào Decoder
interpreter_decoder.set_tensor(decoder_input_details[0]['index'], latent_vector)
interpreter_decoder.invoke()

# Lấy ảnh tái tạo từ Decoder
reconstructed_image = interpreter_decoder.get_tensor(decoder_output_details[0]['index'])

end_time = time.perf_counter()

# Tính thời gian inference
inference_time = end_time - start_time

# Chuyển đổi thành giờ, phút, giây
hours = int(inference_time // 3600)
minutes = int((inference_time % 3600) // 60)
seconds = inference_time % 60  # Giữ phần thập phân cho giây

# 📝 In kết quả
print(f"Thời gian suy luận: {hours} giờ, {minutes} phút, {seconds:.3f} giây")
print(f"Trung bình mỗi ảnh: {inference_time:.3f} giây")

# Hiển thị ảnh gốc và ảnh tái tạo từ mô hình
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Ảnh gốc
axes[0].imshow(x_print[0], cmap='gray')  # Nếu ảnh là grayscale, sử dụng cmap='gray'
axes[0].set_title("Ảnh gốc")
axes[0].axis("off")

# Ảnh tái tạo
axes[1].imshow(reconstructed_image[0], cmap='gray')  # Nếu ảnh là grayscale, sử dụng cmap='gray'
axes[1].set_title("Ảnh tái tạo")
axes[1].axis("off")

plt.show()
