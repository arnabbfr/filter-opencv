import cv2

#loading a pre trained face detection model using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#loading an image we want to use as filter
sunglasses = cv2.imread("sunglasses_PNG.png", cv2.IMREAD_UNCHANGED)

#starting the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

#converting to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecting the faces 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        #resizing the sunglasses to fit the face width
        overlay = cv2.resize(sunglasses,(w, int(h/3)))

        #defining where to place the sunglasses
        y1, y2 = y + int(h/4), y + int(h/4) + overlay.shape[0]
        x1, x2 = x, x + overlay.shape[1]

        #ensuring the overlay does not go out of frame bounds
        if y2 > frame.shape[0] or x2 > frame.shape[1]:
            continue

        #extracting the region of interest from the frame
        roi = frame[y1:y2, x1:x2]

        #separaing the overlay channels
        overlay_rgb = overlay[:, :, :3]
        mask = overlay[:, :, 3]/255.0 # alpha channel as mask

        # Make mask 3-channel to match RGB images
        mask = cv2.merge([mask, mask, mask])

        # Blend overlay with the frame
        roi[:] = (1.0 - mask) * roi + mask * overlay_rgb

     # Show the result
    cv2.imshow("Cool Face Filter 😎", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()