import numpy as np
from objective import calculate_mse, calculate_sse_delta
from stroke import Stroke

class HillClimbingOptimizer:
    def __init__(self, canvas, iterations_per_stroke=50):
        self.canvas = canvas
        self.iters = iterations_per_stroke
        
        # error inicial
        self.current_error = calculate_mse(self.canvas.target_array, self.canvas.current_array)
        print(f"  Error inicial: {self.current_error:.2f}")

    def _get_max_size_for_progress(self, progress):
        w, h = self.canvas.width, self.canvas.height
        max_possible = max(w, h) * 0.3 
        min_possible = max(3.0, max(w, h) * 0.01)

        current_max = max_possible * ((min_possible / max_possible) ** progress)
        return current_max

    def _smart_init_stroke(self, progress):

        w, h = self.canvas.width, self.canvas.height
        

        diff = np.abs(self.canvas.target_array.astype(np.float32) - 
                      self.canvas.current_array.astype(np.float32))
        error_map = diff.mean(axis=2)  # promedio rgb
        

        flat = error_map.flatten()
        total = flat.sum()
        if total == 0:
            return Stroke(w, h)
        
        probs = flat / total
        idx = np.random.choice(len(probs), p=probs) #pixel random con mas error
        cy, cx = divmod(idx, w) #coordenadas del pixel
        
        max_size = self._get_max_size_for_progress(progress) #maximo segun progreso
        s = Stroke(w, h, max_size=max_size) #hacemos el cuadrado
        s.x = cx
        s.y = cy
        

        half = int(s.size / 2)
        y1 = max(0, int(cy) - half)
        y2 = min(h, int(cy) + half)
        x1 = max(0, int(cx) - half)
        x2 = min(w, int(cx) + half)
        
        if x2 > x1 and y2 > y1:
            region = self.canvas.target_array[y1:y2, x1:x2]
            avg = region.mean(axis=(0,1))
            s.r = int(avg[0])
            s.g = int(avg[1])
            s.b = int(avg[2])
        
        return s

    def optimize_next_stroke(self, progress=0.0):

        w, h = self.canvas.width, self.canvas.height
        max_size = self._get_max_size_for_progress(progress)
        

        best_stroke = self._smart_init_stroke(progress)
        

        preview, bbox = self.canvas.preview_stroke_array(best_stroke)
        if bbox is None:
            return None
        

        best_delta = calculate_sse_delta(self.canvas.target_array, self.canvas.current_array, preview, bbox)
        

        for _ in range(self.iters):
            candidate = best_stroke.mutate(w, h, max_size)
            preview, bbox = self.canvas.preview_stroke_array(candidate)
            if bbox is None:
                continue
            
            candidate_delta = calculate_sse_delta(self.canvas.target_array, self.canvas.current_array, preview, bbox)
            

            if candidate_delta < best_delta:
                best_stroke = candidate
                best_delta = candidate_delta
                
        if best_delta < 0:

            preview_final, _ = self.canvas.preview_stroke_array(best_stroke)
            self.current_error = calculate_mse(self.canvas.target_array, preview_final)
            return best_stroke
        else:
            return None
