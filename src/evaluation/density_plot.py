import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_density(input_csv: str, output_dir: str = "outputs/figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Count detections per frame per class
    grouped = df.groupby(["frame_idx", "class_name"]).size().unstack(fill_value=0)

    # Plot 1: All classes over time
    fig, ax = plt.subplots(figsize=(12, 5))
    for cls in grouped.columns:
        ax.plot(grouped.index, grouped[cls], label=cls)
    ax.set_title("Vehicle Density Over Time (per class)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "density_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()

    # Plot 2: Total vehicle count over time
    fig, ax = plt.subplots(figsize=(12, 5))
    total = grouped.drop(columns=["background"], errors="ignore").sum(axis=1)
    ax.fill_between(total.index, total.values, alpha=0.4, color="steelblue")
    ax.plot(total.index, total.values, color="steelblue")
    ax.set_title("Total Vehicle Density Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Total Count")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "density_plot_total.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to frame_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()
    plot_density(args.input, args.output_dir)