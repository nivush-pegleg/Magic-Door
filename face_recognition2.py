import time
import face_recognition
import numpy as np

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []

# Paths to the images of known faces
known_faces_paths = [[r"known_faces\niv\p"+str(i) + ".jpg" for i in range(1, 8)],
                     [r"known_faces\itai\p"+str(i) + ".jpg" for i in range(1, 8)],
                     [r"known_faces\benh\p"+str(i) + ".jpg" for i in range(1, 8)]]

# Add paths for other known faces here

# Load and encode known faces
for face_paths in known_faces_paths:
    face_encodings = []
    for face_path in face_paths:
        face_image = face_recognition.load_image_file(face_path)
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) > 0:
            face_encodings.append(face_encoding[0])
    if len(face_encodings) > 0:
        known_face_encodings.append(np.mean(face_encodings, axis=0))
        known_face_names.append(face_paths[0])  # Assuming the first path is sufficient for naming

# Load the new image with the unknown face
unknown_image = face_recognition.load_image_file(r"try_faces\feb27_faces\p11.jpg")
unknown_face_encodings = face_recognition.face_encodings(unknown_image)
a = time.time()

if len(unknown_face_encodings) > 0:
    # Compare the unknown face with the known faces
    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        print(matches)
        # Check if there is a match
        if True in matches:
            matched_names = [known_face_names[i] for i, match in enumerate(matches) if match]
            print("Found Match:", matched_names)
            b = time.time()
            print(b-a)
            break
        else:
            b = time.time()
            print(b-a)
            print("No match found")
else:
    print("No face found in the unknown image.")
