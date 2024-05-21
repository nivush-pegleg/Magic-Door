import cv2
from deepface import DeepFace
import time

def find_all_faces_in_photo(image_path='reference2.jpg'):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(x, y, w, h)
    # Display the output

    cv2.imshow('img', cv2.resize(img, (960, 540)))
    cv2.waitKey()


def save_faces_in_photo(image_path='reference2.jpg'):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Read the input image
    img = cv2.imread(image_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    face_images = []
    for (x, y, w, h) in faces:
        face_images.append(img[y:y+h, x:x+w])
    # Display the output
    for i, face_image in enumerate(face_images):
        cv2.imwrite(f"frame_{i}.jpg", face_image)
    cv2.waitKey()


def check_face(path_frame1, path_frame2):
    global face_match
    a = time.time()
    frame1 = cv2.imread(path_frame1)
    frame2 = cv2.imread(path_frame2)
    try:
        result = DeepFace.verify(frame1, frame2, model_name='Facenet')
        b = time.time()
        print(b - a)
        if result['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False
    return face_match


def do_faces_match(image1, image2):
    pass


if __name__ == '__main__':
    print(check_face('frame_1.jpg', 'reference.jpg'))