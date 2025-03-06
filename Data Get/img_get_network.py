#########################   Instruction   ##########################
#####  Used to capture images from your camera by using OpenCV  ####
####################################################################

import cv2
import os

def capture_images(output_dir, max_images, frame_interval):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera open failed！")
        return

    print("Camera opened. Start capturing img ...")

    image_count = 0
    frame_count = 0

    while image_count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read video frames!")
            break

        cv2.imshow("on Live", frame)

        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_dir, f"IMG_{image_count:04d}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            image_count += 1

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Pause")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Input 'dog' or 'cat': ")
    scanner = input("dog")
    capture_images(output_dir=os.path.join("../data/", scanner), max_images=1000, frame_interval=1)