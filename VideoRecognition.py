from deepface import DeepFace
import pyautogui
import feed_info
import pathlib
import cv2

def facial_recognition():
    faceCascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(feed_info.url)

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            cv2.imwrite('img.jpg', frame)

            result = DeepFace.verify(img1_path = "img.jpg", 
                    img2_path = "reference.jpg", 
                    enforce_detection=False, 
                    model_name = 'Facenet'
            )

            if result['verified']:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                notification()

        cv2.imshow(feed_info.name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def notification():
    path = pathlib.Path(__file__).parent.resolve()
    img = cv2.imread(f'{path}\\img.jpg',0)
    cv2.imshow('Face detected', img)

    pyautogui.press('stop')

facial_recognition()