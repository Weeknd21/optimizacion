import numpy as np

def calculate_mse(target_arr, canvas_arr):

    diff = target_arr.astype(np.float32) - canvas_arr.astype(np.float32)
    return np.mean(diff * diff)

def calculate_mse_region(target_arr, canvas_arr, bbox):

    x1, y1, x2, y2 = bbox
    t_region = target_arr[y1:y2, x1:x2].astype(np.float32)
    c_region = canvas_arr[y1:y2, x1:x2].astype(np.float32)
    diff = t_region - c_region
    return np.mean(diff * diff)

def calculate_sse_delta(target_arr, canvas_arr, preview_arr, bbox):

    x1, y1, x2, y2 = bbox
    t_region = target_arr[y1:y2, x1:x2].astype(np.float32)
    c_region = canvas_arr[y1:y2, x1:x2].astype(np.float32)
    p_region = preview_arr[y1:y2, x1:x2].astype(np.float32)
    

    diff_old = t_region - c_region
    sse_old = np.sum(diff_old * diff_old)
    

    diff_new = t_region - p_region
    sse_new = np.sum(diff_new * diff_new)
    

    return sse_new - sse_old
