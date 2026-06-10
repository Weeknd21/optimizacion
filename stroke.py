import random
import math
import numpy as np
from PIL import Image, ImageDraw

class Stroke:

    
    def __init__(self, max_x, max_y, max_size=None):
        if max_size is None:
            max_size = max(max_x, max_y) * 0.3
        
        self.x = random.uniform(0, max_x)
        self.y = random.uniform(0, max_y)
        self.size = random.uniform(2, max_size)
        self.angle = random.uniform(0, 360)
        self.r = random.randint(0, 255)
        self.g = random.randint(0, 255)
        self.b = random.randint(0, 255)
        self.alpha = random.randint(30, 220)
    
    def clone(self):
        s = Stroke.__new__(Stroke)
        s.x = self.x
        s.y = self.y
        s.size = self.size
        s.angle = self.angle
        s.r = self.r
        s.g = self.g
        s.b = self.b
        s.alpha = self.alpha
        return s

    def mutate(self, max_x, max_y, max_size=None):
        if max_size is None:
            max_size = max(max_x, max_y) * 0.3
            
        new = self.clone()
        
        n_mutations = random.randint(1, 3)
        choices = random.sample(['position', 'size', 'angle', 'color', 'alpha'], n_mutations)
        
        for mutation_type in choices:
            if mutation_type == 'position':
                new.x = np.clip(self.x + random.gauss(0, max_x * 0.05), 0, max_x)
                new.y = np.clip(self.y + random.gauss(0, max_y * 0.05), 0, max_y)
            elif mutation_type == 'size':
                new.size = np.clip(self.size * random.uniform(0.7, 1.3), 1.0, max_size)
            elif mutation_type == 'angle':
                new.angle = (self.angle + random.gauss(0, 20)) % 360
            elif mutation_type == 'color':
                new.r = int(np.clip(self.r + random.gauss(0, 25), 0, 255))
                new.g = int(np.clip(self.g + random.gauss(0, 25), 0, 255))
                new.b = int(np.clip(self.b + random.gauss(0, 25), 0, 255))
            elif mutation_type == 'alpha':
                new.alpha = int(np.clip(self.alpha + random.gauss(0, 20), 10, 250))
                
        return new

    def get_vertices(self):
        half = self.size / 2
        angle = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        rotated = []
        for dx, dy in corners:
            rx = self.x + dx * cos_a - dy * sin_a
            ry = self.y + dx * sin_a + dy * cos_a
            rotated.append((rx, ry))
        return rotated

    def get_bounding_box(self, img_w, img_h):
        verts = self.get_vertices()
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x1 = max(0, int(min(xs)))
        y1 = max(0, int(min(ys)))
        x2 = min(img_w, int(max(xs)) + 1)
        y2 = min(img_h, int(max(ys)) + 1)
        return x1, y1, x2, y2

    def draw(self, image):
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        vertices = self.get_vertices()
        color = (self.r, self.g, self.b, self.alpha)
        draw.polygon(vertices, fill=color)
        
        base = image.convert('RGBA')
        result = Image.alpha_composite(base, overlay)
        return result.convert('RGB')

    def draw_fast(self, img_array, img_w, img_h):
        x1, y1, x2, y2 = self.get_bounding_box(img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            return None
        
        bw = x2 - x1
        bh = y2 - y1
        mini_overlay = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mini_overlay)
        
        vertices = self.get_vertices()
        local_verts = [(vx - x1, vy - y1) for vx, vy in vertices]
        color = (self.r, self.g, self.b, self.alpha)
        draw.polygon(local_verts, fill=color)
        
        overlay_arr = np.array(mini_overlay, dtype=np.float32)
        alpha = overlay_arr[:, :, 3:4] / 255.0
        fg_rgb = overlay_arr[:, :, :3]
        
        region = img_array[y1:y2, x1:x2].astype(np.float32)
        blended = region * (1 - alpha) + fg_rgb * alpha
        img_array[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return (x1, y1, x2, y2)
