import numpy as np
from PIL import Image

class Canvas:
    def __init__(self, target_image):
        self.target = target_image.convert('RGB')
        self.width, self.height = self.target.size
        self.target_array = np.array(self.target, dtype=np.uint8)
        
        avg_color = tuple(self.target_array.mean(axis=(0,1)).astype(int))
        self.current_image = Image.new('RGB', (self.width, self.height), avg_color)
        self.current_array = np.full_like(self.target_array, avg_color)
        
    def add_stroke(self, stroke):
        stroke.draw_fast(self.current_array, self.width, self.height)
        self.current_image = Image.fromarray(self.current_array)
        
    def preview_stroke_array(self, stroke):
        preview = self.current_array.copy()
        bbox = stroke.draw_fast(preview, self.width, self.height)
        return preview, bbox

    def get_image(self):
        return Image.fromarray(self.current_array)
