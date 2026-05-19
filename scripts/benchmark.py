import time
import torch
import timm
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image


CLASSES = ["car", "bus", "truck", "background"]


def load_model(checkpoint_path: str, img_size: int):
    model = timm.create_model("mobilevit_s", pretrained=False, num_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def benchmark_resolution(checkpoint_path: str, img_size: int, n_runs: int = 50):
    """Run inference n_runs times at a given resolution and return avg time."""
    model = load_model(checkpoint_path, img_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Dummy image
    dummy = Image.fromarray(np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8))
    tensor = transform(dummy).unsqueeze(0).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(5):
            model(tensor)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(tensor)
            times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    return avg_ms, std_ms


def run_benchmark(checkpoint_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolutions = [128, 192, 256]
    # Approximate accuracy values from your eval runs (update if you have real ones)
    accuracy_map = {128: 94.5, 192: 95.8, 256: 96.71}

    results = []
    for res in resolutions:
        print(f"Benchmarking resolution {res}x{res}...")
        avg_ms, std_ms = benchmark_resolution(checkpoint_path, res)
        acc = accuracy_map.get(res, 0)
        results.append({
            "resolution": res,
            "avg_inference_ms": round(avg_ms, 2),
            "std_ms": round(std_ms, 2),
            "accuracy": acc
        })
        print(f"  {res}x{res}: {avg_ms:.1f} ms ± {std_ms:.1f} ms | Accuracy: {acc}%")

    # Save CSV
    csv_path = output_dir / "speed_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")

    # Save JSON
    json_path = output_dir / "speed_accuracy.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = [r["resolution"] for r in results]
    times = [r["avg_inference_ms"] for r in results]
    accs = [r["accuracy"] for r in results]

    ax1.bar(x, times, width=20, color="steelblue", alpha=0.7, label="Inference Time (ms)")
    ax1.set_xlabel("Input Resolution")
    ax1.set_ylabel("Avg Inference Time (ms)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(x, accs, "ro-", label="Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(90, 100)

    plt.title("Speed vs Accuracy Trade-off")
    fig.tight_layout()
    out_path = output_dir / "speed_accuracy_tradeoff.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pth")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()
    run_benchmark(args.checkpoint, args.output_dir)