#########################   Instruction   ##########################
#####              Process and classify new photos              ####
####################################################################

import numpy as np
import mlx.core as mx
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path).resize((224, 224))

    img_array = np.array(img).astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std

    img_array = np.expand_dims(img_array, axis=0)
    return mx.array(img_array)

def predict(model, image_path):
    test_image = load_image(image_path)

    logits = model(test_image)

    probabilities = mx.softmax(logits, axis=1)

    predicted_class = mx.argmax(probabilities, axis=1).item()

    class_names = {0: "dog", 1: "cat"}
    return class_names[predicted_class]