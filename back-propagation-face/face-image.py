import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('hiphop.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(
img, "naveen", (x, y), font, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
cv2.imshow('mahesh output', img)
cv2.waitKey()
