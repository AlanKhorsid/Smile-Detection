import cv2

print("Package Imported")

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

print('[INFO] starting smile detection...')
webcam = cv2.VideoCapture(0)  # 0 is the default webcam


def visualise_smiles(frame, rect_text, x, y, w, h, b, g, r, line_thickness, add_rect):
    if add_rect is True:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (b, g, r), 4)
    cv2.putText(frame, rect_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (b, g, r), line_thickness, cv2.LINE_AA)


def smile_detection(frame, faces, smiles, monkey_smiles):
    for (x, y, w, h) in faces:
        if len(smiles) == 0:
            visualise_smiles(frame, "Not Smiling", x, y,
                             w, h, 0, 0, 255, 2, True)

        for smile in smiles:
            if smiles is not None:
                for ms in monkey_smiles:
                    visualise_smiles(frame, "Monkey",
                                     x+130, y, w, h, 0, 255, 0, 2, False)

                visualise_smiles(frame, "Smiling", x, y, w,
                                 h, 49, 228, 21, 2, True)
            else:
                visualise_smiles(frame, "Not Smiling", x,
                                 y, w, h, 0, 0, 255, 2, True)


while True:
    success, frame = webcam.read()  # Reads the frame from the webcam
    if not success:
        break

    # Converts the frame to greyscale for easier processing
    frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        frame_greyscale,
        scaleFactor=1.7,
        minNeighbors=5
    )  # Detects the faces in the frame
    smiles = smile_detector.detectMultiScale(
        frame_greyscale,
        scaleFactor=1.7,
        minNeighbors=40
    )  # Detects the smiles in the frame
    monkey_smiles = smile_detector.detectMultiScale(
        frame_greyscale,
        scaleFactor=1.7,
        minNeighbors=70
    )

    smile_detection(frame, faces, smiles, monkey_smiles)

    cv2.imshow("Smile Detector", frame)  # Displays the frame
    cv2.waitKey(1)  # Waits for 1 millisecond

webcam.release(0)  # Releases the webcam
cv2.destroyAllWindows("Smile Detector")  # Destroys all windows
