import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Loop through each frame in the webcam feed
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's face detection (Haar cascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # Draw a rectangle on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Perform face recognition using DeepFace
        try:
            result = DeepFace.analyze(face, actions=['age', 'gender', 'race', 'emotion'])
            print(result)
            
            # Add the result to the frame
            cv2.putText(frame, f"Gender: {result['gender']}, Age: {result['age']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.putText(frame, f"Emotion: {result['dominant_emotion']}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        except Exception as e:
            print(f"Error: {e}")
    
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
