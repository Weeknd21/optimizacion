import numpy as np
from objective import calculate_mse, calculate_mse_region
from stroke import Stroke

class HillClimbingOptimizer:
    def __init__(self, canvas, iterations_per_stroke=50):
        self.canvas = canvas
        self.iters = iterations_per_stroke
        
        # Error global actual
        self.current_error = calculate_mse(self.canvas.target_array, self.canvas.current_array)
        print(f"  Error inicial: {self.current_error:.2f}")

    def _get_max_size_for_progress(self, progress):
        """
        Reduce el tamaño maximo del stroke conforme avanza el proceso.
        Empieza permitiendo strokes del 30% del tamaño de la imagen,
        y termina permitiendo solo strokes muy pequeños (ej. 2% o 3 pixeles).
        """
        w, h = self.canvas.width, self.canvas.height
        max_possible = max(w, h) * 0.3
        min_possible = max(3.0, max(w, h) * 0.01)
        
        # Decrecimiento exponencial del tamaño
        current_max = max_possible * ((min_possible / max_possible) ** progress)
        return current_max

    def _smart_init_stroke(self, progress):
        """
        Inicializa un stroke de forma inteligente:
        coloca el cuadrado donde hay mas error y le asigna
        el color promedio de esa zona en la imagen objetivo.
        """
        w, h = self.canvas.width, self.canvas.height
        
        # Calcular mapa de error por pixel
        diff = np.abs(self.canvas.target_array.astype(np.float32) - 
                      self.canvas.current_array.astype(np.float32))
        error_map = diff.mean(axis=2)  # Promediar canales RGB
        
        # Elegir una posicion con probabilidad proporcional al error
        flat = error_map.flatten()
        total = flat.sum()
        if total == 0:
            return Stroke(w, h)
        
        probs = flat / total
        idx = np.random.choice(len(probs), p=probs)
        cy, cx = divmod(idx, w)
        
        max_size = self._get_max_size_for_progress(progress)
        s = Stroke(w, h, max_size=max_size)
        s.x = cx
        s.y = cy
        
        # Asignar el color promedio de la region objetivo alrededor de ese punto
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
        """
        Genera y optimiza un nuevo stroke usando Hill Climbing.
        Usa MSE local (solo en la zona del stroke) para ser rapido.
        """
        w, h = self.canvas.width, self.canvas.height
        max_size = self._get_max_size_for_progress(progress)
        
        # 1. Inicializar stroke inteligentemente (donde hay mas error)
        best_stroke = self._smart_init_stroke(progress)
        
        # Evaluar el stroke candidato
        preview, bbox = self.canvas.preview_stroke_array(best_stroke)
        if bbox is None:
            return None
        best_error = calculate_mse(self.canvas.target_array, preview)
        
        # 2. Hill climbing: mutar y mejorar
        for _ in range(self.iters):
            candidate = best_stroke.mutate(w, h, max_size)
            preview, bbox = self.canvas.preview_stroke_array(candidate)
            if bbox is None:
                continue
            candidate_error = calculate_mse(self.canvas.target_array, preview)
            
            if candidate_error < best_error:
                best_stroke = candidate
                best_error = candidate_error
                
        # 3. Solo aceptar si mejora el error global
        if best_error < self.current_error:
            self.current_error = best_error
            return best_stroke
        else:
            return None
