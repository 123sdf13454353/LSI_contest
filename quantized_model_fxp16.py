from fixedpoint_update import convert_weights_to_fxp ,extract_weights_recursive,save_weights_to_dict,update_weights_in_model
from config import encoder_floatpoint,decoder_floatpoint
from model import VAE

from tensorflow.keras.models import load_model

vae_floatpoint=VAE(encoder_floatpoint,decoder_floatpoint)


weights_dict_encoder = save_weights_to_dict(vae_floatpoint.encoder)
print("🔍 Trọng số ban đầu đã trích xuất.")


fxp_weights_dict_encoder = convert_weights_to_fxp(weights_dict_encoder)
print("✅ Trọng số đã được chuyển sang fixed-point.")


update_weights_in_model(vae_floatpoint.encoder, fxp_weights_dict_encoder)
print("✅ Mô hình đã được cập nhật với trọng số fixed-point.")

# Bước 5: Lưu mô hình sau khi cập nhật trọng số fixed-point
vae_floatpoint.encoder.save("vae_encoder_fixedpoint.h5")
print("✅ Mô hình đã được lưu với trọng số fixed-point.")

#quantized bộ decoder

weights_dict_decoder = save_weights_to_dict(vae_floatpoint.decoder)
print("🔍 Trọng số ban đầu đã trích xuất.")


fxp_weights_dict_decoder = convert_weights_to_fxp(weights_dict_decoder)
print("✅ Trọng số đã được chuyển sang fixed-point.")


update_weights_in_model(vae_floatpoint.decoder, fxp_weights_dict_decoder)
print("✅ Mô hình đã được cập nhật với trọng số fixed-point.")

# Bước 5: Lưu mô hình sau khi cập nhật trọng số fixed-point
vae_floatpoint.decoder.save("vae_decoder_fixedpoint.h5")


