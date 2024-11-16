import cv2
import mediapipe as mp

from gesture_copy import mp_drawing

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def classify_gesture(landmarks):

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP].y

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y

    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP].y

    is_thumb_extended = thumb_tip < thumb_ip
    is_index_extended = index_tip < index_mcp
    is_middle_extended = middle_tip < middle_mcp
    is_ring_extended = ring_tip < ring_mcp
    is_pinky_extended = pinky_tip < pinky_mcp

    if is_thumb_extended and is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended:
        return 'Stop'
    elif is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
        return 'One'
    elif is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
        return 'Peace'
    elif is_index_extended and is_middle_extended and is_ring_extended and not is_pinky_extended:
        return 'Three'
    elif is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended and not is_thumb_extended:
        return 'Four'
    else:
        return 'Unknown'

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
      ret , frame = cap.read()
      if not ret:
          break

      frame = cv2.flip(frame ,1)
      frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

      result = hands.process(frame_rgb)

      if result.multi_hand_landmarks:
          for hand_landmarks in result.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                  frame,hand_landmarks,mp.HAND_CONNECTIONS
              )

              gesture = classify_gesture(hand_landmarks.landmark)

              cv2.putText(frame, f"Gesture : {gesture}" , (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv2.LINE_AA)

      cv2.imshow("GESTURE RECOGNITION" , frame)

      if cv2.waitKey(1) & 0xFF == 27:
          break

cap.release()
cv2.destroyAllWindows()






