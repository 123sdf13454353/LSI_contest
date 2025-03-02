import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from dataset import load_dataset
from model import build_encoder, build_decoder, VAE
from config import MODEL_PATH, ENCODER_PATH, DECODER_PATH, EPOCHS



# Load dataset
x_train_new, x_test = load_dataset()

# Xây dựng mô hình
encoder = build_encoder()
decoder = build_decoder()
vae = VAE(encoder, decoder, beta=0.001)

vae.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

# Callbacks
callbacks_list = [
    ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

# Huấn luyện
vae.fit(x=x_train_new, y=x_train_new, epochs=EPOCHS, validation_split=0.2, callbacks=callbacks_list, verbose=1)

# Lưu mô hình
vae.encoder.save(ENCODER_PATH)
vae.decoder.save(DECODER_PATH)



