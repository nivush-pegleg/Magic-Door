import cv2
from time import time
from face_recognition4 import predict


def image_cap(folder_path, face_encodings):
    record = False

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to read camera feed")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        k = cv2.waitKey(1)

        if ret:
            cv2.imshow('frame', frame)

            # press s key to start recording
            if k & 0xFF == ord('s'):
                record = True

            if record:
                filename = "frame_" + str(int(time())) + ".jpg"
                new_pic_path = folder_path + r"/" + filename
                cv2.imwrite(new_pic_path, frame)
                predicted_label, confidence = predict(new_pic_path, face_encodings)
                print("Predicted Label:", predicted_label)
                print("Similarity ratio:", confidence)

                # press q key to close the program
            if k & 0xFF == ord('q'):
                break
            record = False

        else:
            break

    cap.release()
    out.release()

    cv2.destroyAllWindows()
