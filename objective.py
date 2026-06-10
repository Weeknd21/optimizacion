import numpy as np

def calculate_mse(target_arr, canvas_arr):
    """
    Calcula el Mean Squared Error (MSE) entre dos arrays numpy uint8.
    """
    diff = target_arr.astype(np.float32) - canvas_arr.astype(np.float32)
    return np.mean(diff * diff)

def calculate_mse_region(target_arr, canvas_arr, bbox):
    """
    Calcula el MSE solo en una region (bounding box) de la imagen.
    bbox = (x1, y1, x2, y2)
    Esto es MUCHO mas rapido que calcular el MSE de toda la imagen.
    """
    x1, y1, x2, y2 = bbox
    t_region = target_arr[y1:y2, x1:x2].astype(np.float32)
    c_region = canvas_arr[y1:y2, x1:x2].astype(np.float32)
    diff = t_region - c_region
    return np.mean(diff * diff)
