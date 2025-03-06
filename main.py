#########################   Instruction   ##########################
#####                          Main                             ####
####################################################################

import os
from predict import predict
from train import train_model

if __name__ == "__main__":
    print("Model training starts...")
    trained_model = train_model()

    test_image_path = [
        os.path.join("./Photos", fname)
        for fname in os.listdir("./Photos")
        if os.path.splitext(fname)[1].lower() in (".jpg", ".jpeg", ".png")
    ]

    print("Image classification starts...")
    for image_path in test_image_path:
        predicted_class = predict(trained_model, image_path)
        print(f"Image: {image_path} -> Predicted class: {predicted_class}")