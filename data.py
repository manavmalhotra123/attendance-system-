import cv2
import os

# Create the "manav" folder if it does not exist
if not os.path.exists('ayush'):
    os.makedirs('ayush')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Initialize frame counter
frame_num = 0

# Loop through each frame in the video stream
while True:
    # Read the frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Save each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) from the original image
        roi = frame[y:y+h, x:x+w]
        # Save the ROI with a unique name
        filename = 'face-{}.jpg'.format(frame_num)
        cv2.imwrite(os.path.join('ayush', filename), roi)
        frame_num += 1

        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame with bounding boxes
    cv2.imshow('Face Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()