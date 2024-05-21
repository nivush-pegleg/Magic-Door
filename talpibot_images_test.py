from save_image_from_driver import image_cap
from face_recognition4 import predict
from face_recognition4 import *
import os


known_im_folder = "known_faces"

# Prepare training data
face_encodings, labels = prepare_training_data(known_im_folder)
try_images_path = r"talpibot_images/all_together"
print("start")
files = [f for f in os.listdir(try_images_path) if os.path.isfile(os.path.join(try_images_path, f))]
print(len(files))
for file in files:
    print(file, ":", predict(try_images_path + "/" + file, face_encodings))
