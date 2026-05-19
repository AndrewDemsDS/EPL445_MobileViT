import cv2
import torch
import timm
import numpy as np
from torchvision import transforms
from pathlib import Path


CLASSES = ["car", "bus", "truck", "background"]

COLORS = {
    "car":        (0, 255, 0),
    "bus":        (255, 0, 0),
    "truck":      (0, 0, 255),
    "background": (128, 128, 128),
}


def load_model(checkpoint_path: str, device: torch.device):
    model = timm.create_model("mobilevit_s", pretrained=False, num_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model.to(device)


def get_transform(img_size: int = 256):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def run_stream(source, checkpoint_path: str, img_size: int = 256, window_size: int = 64, stride: int = 32):
    """
    Run real-time inference on a video file, webcam, or RTSP stream.

    source: 
      - int (0, 1, ...) for webcam
      - str path to video file
      - str "rtsp://..." for RTSP stream
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(checkpoint_path, device)
    transform = get_transform(img_size)

    # Open source
    if isinstance(source, str) and source.startswith("rtsp://"):
        print(f"Connecting to RTSP stream: {source}")
    elif isinstance(source, int):
        print(f"Opening webcam {source}")
    else:
        print(f"Opening video file: {source}")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    print("Stream opened. Press 'q' to quit.")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or no frame received.")
            break

        h, w = frame.shape[:2]
        counts = {cls: 0 for cls in CLASSES}

        # Sliding window inference
        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                patch = frame[y:y+window_size, x:x+window_size]
                tensor = transform(patch).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    pred = torch.argmax(output, dim=1).item()
                    label = CLASSES[pred]

                if label != "background":
                    counts[label] += 1
                    color = COLORS[label]
                    cv2.rectangle(frame, (x, y), (x+window_size, y+window_size), color, 1)
                    cv2.putText(frame, label, (x, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Overlay counts
        y_offset = 20
        for cls, count in counts.items():
            if cls == "background":
                continue
            text = f"{cls}: {count}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
            y_offset += 25

        cv2.putText(frame, f"Frame: {frame_id}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Traffic Stream", frame)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0=webcam, path to video, or rtsp://...")
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pth")
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    # Convert to int if webcam index
    source = int(args.source) if args.source.isdigit() else args.source
    run_stream(source, args.checkpoint, args.img_size)