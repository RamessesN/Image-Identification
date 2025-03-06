#########################   Instruction   ##########################
#####        Used to pre-operate images that captured           ####
####################################################################

import random
import numpy as np
import mlx.core as mx
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

def files_and_classes(root: Path):
    classes = {"dog": 0, "cat": 1}
    files = []
    for class_name, label in classes.items():
        class_dir = root / class_name
        if class_dir.exists():
            class_files = list(class_dir.glob("*.jpg"))
            print(f"Class {class_name}: {len(class_files)} samples")
            files.extend([{"image": str(f), "label": label} for f in class_files])
    return files

def random_horizontal_flip(image):
    """ Horizontal flip randomly """
    if random.random() < 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def random_rotation(image, max_angle=10):
    """ Rotate randomly by ±max_angle degrees """
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, resample=Image.BILINEAR)

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """ Color jitter with corrected hue adjustment """
    # Brightness
    factor = 1.0 + random.uniform(-brightness, brightness)
    image = ImageEnhance.Brightness(image).enhance(factor)

    # Contrast
    factor = 1.0 + random.uniform(-contrast, contrast)
    image = ImageEnhance.Contrast(image).enhance(factor)

    # Saturation
    factor = 1.0 + random.uniform(-saturation, saturation)
    image = ImageEnhance.Color(image).enhance(factor)

    # Hue (approximation using colorize)
    if random.random() < 0.5:
        image = ImageOps.colorize(image.convert("L"), black="blue", white="red")

    return image

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format

    # Data augmentation
    img = random_horizontal_flip(img)
    img = random_rotation(img, max_angle=10)
    img = color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    # Resize and convert to array
    img = img.resize((128, 128))
    img_array = np.array(img).astype("float32") / 255.0

    # Normalize
    mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std

    # Ensure channel order is correct (HWC -> CHW)
    img_array = np.transpose(img_array, (2, 0, 1))  # (height, width, channels) -> (channels, height, width)

    return img_array

def load_dataset(root, batch_size):
    dataset = files_and_classes(root)
    batches = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        images = np.array([load_image(item["image"]) for item in batch])
        labels = np.array([item["label"] for item in batch])

        images = mx.array(images)
        labels = mx.array(labels)

        batches.append({"image": images, "label": labels})
    return batches