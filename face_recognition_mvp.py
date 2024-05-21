from save_image_from_driver import image_cap
from face_recognition4 import *

known_im_folder = "known_faces"

# Prepare training data
face_encodings, labels = prepare_training_data(known_im_folder)

try_image_path = r"try_faces/blitz_faces"
print("start")
image_cap(try_image_path, face_encodings)
