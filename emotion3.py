import cv2
import time
from deepface import DeepFace
import numpy as np

# Load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

start_time = time.time()
emotion_scores = {emotion: 0 for emotion in emotion_labels}
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = gray_frame[y:y + h, x:x + w]

        # Resize the face ROI to match the input shape of the model
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the resized face image
        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # Map the emotion prediction scores to a range of 1 to 100
        emotion_level = int(preds[emotion_idx] * 100)

        # Increment the emotion score
        emotion_scores[emotion] += emotion_level

        frame_count += 1

        # Draw rectangle around face and label with predicted emotion and emotion level
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"{emotion} ({emotion_level})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) >= 10:
        break

# Calculate and print the average emotion
average_emotion = max(emotion_scores, key=emotion_scores.get)

# Sort the emotion scores in descending order
sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

# Get the emotion that was predicted the second most
second_max_emotion = sorted_emotions[1][0]

# Print the second most predicted emotion if the average emotion is neutral
if average_emotion == emotion_labels[6] and emotion_scores.get(average_emotion) != frame_count:
    average_emotion_level = emotion_scores[second_max_emotion] / frame_count
    print(f"Average Emotion:  {second_max_emotion} (level: {average_emotion_level:.2f})")
else:
    average_emotion_level = emotion_scores[average_emotion] / frame_count
    print(f"Average Emotion: {average_emotion} (level: {average_emotion_level:.2f})")

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
