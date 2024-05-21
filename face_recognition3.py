import os
import face_recognition


# Function to prepare training data
def prepare_training_data(data_folder_path):
    face_encodings = []
    labels = []

    for person_name in os.listdir(data_folder_path):
        person_dir_path = os.path.join(data_folder_path, person_name)
        print(person_dir_path)

        if not os.path.isdir(person_dir_path):
            continue

        label = person_name
        images = os.listdir(person_dir_path)

        for image_name in images:
            if image_name.startswith("."):
                continue
            image_path = os.path.join(person_dir_path, image_name)
            image = face_recognition.load_image_file(image_path)
            try:
                face_encoding = face_recognition.face_encodings(image)[0]

            except IndexError:
                print("No face found in the image")
                continue

            face_encodings.append(face_encoding)
            labels.append(label)

    return face_encodings, labels


def predict(test_img_path, threshold=0.6):
    test_image = face_recognition.load_image_file(test_img_path)
    test_encoding = face_recognition.face_encodings(test_image)

    if len(test_encoding) == 0:
        return False, 0.0

    max_confidence = 0.0
    for test_face_encoding in test_encoding:
        for known_face_encoding in face_encodings:
            # Compare the faces
            match = face_recognition.compare_faces([known_face_encoding], test_face_encoding, tolerance=threshold)
            if match[0]:
                # Calculate the confidence level
                face_distances = face_recognition.face_distance([known_face_encoding], test_face_encoding)
                confidence = 1 - face_distances[0]
                if confidence > max_confidence:
                    max_confidence = confidence

    if max_confidence > 0.0:
        return True, max_confidence
    else:
        return False, 0.0


# Path to the images folder
images_folder = "known_faces"

# Prepare training data
face_encodings, labels = prepare_training_data(images_folder)

test_img_path = r"try_faces\blitz_faces\p15.jpg"
predicted_label, confidence = predict(test_img_path)
print("Predicted Label:", predicted_label)
print("Confidence:", confidence)
