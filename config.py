from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from fxpmath import Fxp

import os

# Cấu hình dữ liệu
IMAGE_ROOT = "C:/LSI/hazelnut"
IMAGE_SIZE = (64, 64)

# Cấu hình VAE
LATENT_DIM = 32
EPOCHS = 200
MODEL_PATH = "capsule.keras"

# Đường dẫn lưu mô hình
ENCODER_PATH = "vae_encoder_float.h5"
DECODER_PATH = "vae_decoder_float.h5"

encoder_floatpoint=load_model(ENCODER_PATH)
decoder_floatpoint=load_model(DECODER_PATH)
# vae_floatpoint=VAE(encoder_floatpoint,decoder_floatpoint)

# ENCODER_FIXED_PATH= "vae_encoder_fixed_16_8_fixed_point.h5"
# DECODER_FIXED_PATH= "vae_encoder_fixed_16_8_fixed_point.h5" 

# encoder_fixedpoint=load_model(ENCODER_FIXED_PATH)
# decoder_fixedpoint=load_model(DECODER_FIXED_PATH)

# vae_fixedpoint=VAE(encoder_fixedpoint,decoder_fixedpoint)
