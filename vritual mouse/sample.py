import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Create a video capture object
cap = cv2.VideoCapture(0)

# Initialize variables
drawing = False
ix, iy = -1, -1

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # If hand landmarks are detected, draw something based on finger movements
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks
            landmarks = hand_landmarks.landmark

            # Extract specific landmarks for the fingers
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

            # Check the y-coordinate of the finger tips
            if thumb_tip.y < index_tip.y and middle_tip.y < ring_tip.y and ring_tip.y < pinky_tip.y:
                cv2.putText(frame, 'Thumbs Up!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
