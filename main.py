import os
import sys
import time
import argparse
from PIL import Image
from canvas import Canvas
from optimizer import HillClimbingOptimizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruccion de imagenes mediante strokes (cuadrados) optimizados."
    )
    parser.add_argument(
        "target", nargs="?", default="evngelion.jpg",
        help="Ruta de la imagen objetivo (default: evngelion.jpg)"
    )
    parser.add_argument(
        "-n", "--strokes", type=int, default=3000,
        help="Numero total de strokes a colocar (default: 3000)"
    )
    parser.add_argument(
        "-r", "--resolution", type=int, default=400,
        help="Resolucion maxima de trabajo en pixeles (default: 400)"
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=50,
        help="Iteraciones de hill climbing por stroke (default: 50)"
    )
    parser.add_argument(
        "-o", "--output", default="output",
        help="Directorio de salida (default: output)"
    )
    parser.add_argument(
        "--save-every", type=int, default=25,
        help="Guardar progreso cada N strokes (default: 25)"
    )
    parser.add_argument(
        "--max-fails", type=int, default=80,
        help="Fallos consecutivos antes de parar (default: 80)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Muestra una ventana actualizandose en tiempo real (requiere opencv-python)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.live:
        try:
            import cv2
        except ImportError:
            args.live = False


    if not os.path.isfile(args.target):
        print(f"Error: No se encontro la imagen '{args.target}'")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)


    print(f"Imagen objetivo: {args.target}")
    target_img = Image.open(args.target)
    print(f"  Resolucion original: {target_img.size}")

    target_img.thumbnail((args.resolution, args.resolution), Image.LANCZOS)
    print(f"  Resolucion de trabajo: {target_img.size}")

    canvas = Canvas(target_img)


    target_img.save(os.path.join(args.output, "target_resized.png"))


    optimizer = HillClimbingOptimizer(canvas, iterations_per_stroke=args.iterations)

    print(f"\n{'='*50}")
    print(f"  Strokes objetivo:   {args.strokes}")
    print(f"  Iteraciones/stroke: {args.iterations}")
    print(f"  Guardar cada:       {args.save_every} strokes")
    print(f"  Max fallos:         {args.max_fails}")
    print(f"{'='*50}\n")

    added = 0
    fails = 0
    start_time = time.time()

    while added < args.strokes:

        progress = added / args.strokes
        stroke = optimizer.optimize_next_stroke(progress)

        if stroke is not None:
            canvas.add_stroke(stroke)
            added += 1
            fails = 0

            if args.live:
                bgr_img = cv2.cvtColor(canvas.current_array, cv2.COLOR_RGB2BGR)
                cv2.imshow("Optimizacion", bgr_img)
                cv2.waitKey(1)

            if added % args.save_every == 0:
                elapsed = time.time() - start_time
                rate = added / elapsed if elapsed > 0 else 0
                out_path = os.path.join(args.output, f"progress_{added:04d}.png")
                canvas.get_image().save(out_path)
                print(f"  [{added}/{args.strokes}] Error: {optimizer.current_error:.2f} | "
                      f"{rate:.1f} strokes/seg | {out_path}")
        else:
            fails += 1
            if fails >= args.max_fails:

                out_path = os.path.join(args.output, f"progress_{added:04d}_stalled.png")
                canvas.get_image().save(out_path)
                print(f"\n  Estancado despues de {added} strokes ({args.max_fails} fallos consecutivos).")
                print(f"  Ultimo progreso guardado: {out_path}")
                break

    elapsed = time.time() - start_time


    final_path = os.path.join(args.output, "final_result.png")
    canvas.get_image().save(final_path)

    print(f"\n{'='*50}")
    print(f"  Strokes colocados: {added}")
    print(f"  Tiempo total:      {elapsed:.1f}s")
    print(f"  Velocidad:         {added / elapsed:.1f} strokes/seg" if elapsed > 0 else "")
    print(f"  Error final:       {optimizer.current_error:.2f}")
    print(f"  Resultado:         {final_path}")
    print(f"{'='*50}")

    if args.live:

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
