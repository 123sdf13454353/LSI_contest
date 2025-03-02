from fxpmath import Fxp

# Định nghĩa tham chiếu Fxp
fxp_ref_1 = Fxp(None, dtype='fxp-s16/8')

def convert_weights_to_fxp(w_dict):
    """
    Chuyển đổi weights & bias sang fixed-point sử dụng fxpmath.
    """
    w_fxp_dict = {}
    for layer, params in w_dict.items():
        w_fxp_dict[layer] = [
            Fxp(params["weights"], like=fxp_ref_1) if params["weights"] is not None else None,
            Fxp(params["bias"], like=fxp_ref_1) if params["bias"] is not None else None
        ]
    return w_fxp_dict


def extract_weights_recursive(layer, w_dict):
    """
    Đệ quy duyệt qua các lớp và lưu weights & bias vào w_dict.
    """
    if hasattr(layer, 'layers'):  # Nếu layer chứa lớp con (vd. Sequential)
        for sub_layer in layer.layers:
            extract_weights_recursive(sub_layer, w_dict)
    else:
        weights_and_bias = layer.get_weights()
        if len(weights_and_bias) > 0:  # Nếu có weights và bias
            w_dict[layer.name] = {
                "weights": weights_and_bias[0],
                "bias": weights_and_bias[1] if len(weights_and_bias) > 1 else None
            }
        else:
            w_dict[layer.name] = {"weights": None, "bias": None}

def save_weights_to_dict(model):
    """
    Trích xuất toàn bộ trọng số từ model và lưu vào dictionary.
    """
    w_dict = {}
    extract_weights_recursive(model, w_dict)
    return w_dict



def update_weights_in_model(model, w_fxp_dict):
    """
    Cập nhật mô hình với trọng số và bias đã chuyển đổi.
    """
    for layer_name, values in w_fxp_dict.items():
        layer = model.get_layer(layer_name)

        if hasattr(layer, 'weights') and len(layer.weights) > 0:
            if values[0] is not None and values[1] is not None:
                layer.set_weights([values[0], values[1]])
                print(f"✅ Đã cập nhật trọng số cho lớp: {layer_name}")
        else:
            print(f"⚠️ Lớp {layer_name} không có trọng số hoặc bias để cập nhật.")



