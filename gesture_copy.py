import cv2  # cv2 is used for computer vison
import mediapipe as mp # A library specifically designed for building multimodal applied machine learning pipelines, here used for hand tracking
import webbrowser

# Initialize MediaPipe Hands and drawing utils
mp_hands = mp.solutions.hands  # initialized to manage hand detection and tracking.
mp_drawing = mp.solutions.drawing_utils  # to visualize the detected hand landmarks.


# Hand gesture classifier function using multiple landmarks for more accurate classification
def classify_gesture(landmarks):
    # Get coordinates for each finger tip (4 for thumb, 8 for index, 12 for middle, 16 for ring, 20 for pinky)
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP].y  # Tip and Intermediate Phalanx
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y  # Tip and Metacarpophalangeal (MCP)
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    wrist = landmarks[mp_hands.HandLandmark.WRIST].y

    # Calculate if fingers are extended or not
    is_thumb_extended = thumb_tip < thumb_ip
    is_index_extended = index_tip < index_mcp
    is_middle_extended = middle_tip < middle_mcp
    is_ring_extended = ring_tip < ring_mcp
    is_pinky_extended = pinky_tip < pinky_mcp

    # Classify gestures based on finger positions
    if is_thumb_extended and is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended:
        return 'Stop'  # All fingers extended (open palm)
    elif is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
        return 'Peace'  # Only index and middle fingers extended
    elif not is_thumb_extended and not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
        return 'Punch'  # All fingers curled (fist)
    else:
        return 'Unknown'


# Capture video from the webcam
cap = cv2.VideoCapture(0)

youtube_opened = False
google_opened = False
whatsapp_opened = False

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirrored effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Classify gesture
                gesture = classify_gesture(hand_landmarks.landmark)

                # Display the gesture on the screen
                cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if gesture == 'Stop' and not youtube_opened:
                    webbrowser.open("https://www.youtube.com/")
                    youtube_opened = True

                elif gesture == 'Punch' and not google_opened:
                    webbrowser.open("https://www.google.com/")
                    google_opened = True

                elif gesture == 'Peace' and not whatsapp_opened:
                    webbrowser.open("https://www.whatsapp.com/")
                    whatsapp_opened = True

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()