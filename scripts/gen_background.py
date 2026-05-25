"""Generate background crops from existing car images for the 'background' class."""
import random
from pathlib import Path
from PIL import Image

random.seed(42)
bg_dir = Path("data/processed/background")
bg_dir.mkdir(exist_ok=True)

car_imgs = list(Path("data/processed/car").glob("*.jpg"))
random.shuffle(car_imgs)
car_imgs = car_imgs[:4000]

crop_size, target, count = 64, 2000, 0
for fp in car_imgs:
    if count >= target:
        break
    try:
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        if w < crop_size or h < crop_size:
            continue
        left = random.choice([0, w - crop_size])
        top = random.choice([0, h - crop_size])
        crop = img.crop((left, top, left + crop_size, top + crop_size))
        crop.save(str(bg_dir / f"bg_{fp.stem}_{left}_{top}.jpg"), "JPEG", quality=85)
        count += 1
    except Exception:
        continue

print(f"Generated {count} background crops -> {bg_dir}")
