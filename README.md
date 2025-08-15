# Gesture-Detection-Project
import cv2
import mediapipe as mp

# Mediapipe modules
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Hand detection
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Finger tip landmark IDs
finger_tips = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 1, 0, 0, 0]:
        return "Pointing"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Hand"
    else:
        return "Unknown Gesture"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger state detection
            fingers = []
            lm_list = []
            h, w, _ = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Thumb
            fingers.append(1 if lm_list[finger_tips[0]][0] > lm_list[finger_tips[0] - 1][0] else 0)

            # Other fingers
            for tip in finger_tips[1:]:
                fingers.append(1 if lm_list[tip][1] < lm_list[tip - 2][1] else 0)

            gesture = detect_gesture(fingers)
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("Detected Gesture:", gesture)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


# Code Explanation
1. Importing Libraries
   import cv2
import mediapipe as mp

cv2 → OpenCV for camera input and image display.
mediapipe → Google’s library for hand & pose detection.

2. Mediapipe Modules Setup
   mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

mp_hands → Handles the hand detection model.
mp_draw → Used to draw landmarks & connections on the hand.

3. Hand Detection Model Initialization
   hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

max_num_hands=1 → Detect only one hand.
min_detection_confidence=0.7 → Minimum confidence for detection to be considered valid.
min_tracking_confidence=0.7 → Confidence needed to keep tracking the hand.

4. Finger Tip Landmarks
   finger_tips = [4, 8, 12, 16, 20]

Each number corresponds to a fingertip landmark in Mediapipe's hand model:
4 → Thumb tip
8 → Index finger tip
12 → Middle finger tip
16 → Ring finger tip
20 → Pinky tip

5. Starting the Webcam
   cap = cv2.VideoCapture(0)

Opens the default webcam.

6. Gesture Detection Logic
   def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 1, 0, 0, 0]:
        return "Pointing"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Hand"
    else:
        return "Unknown Gesture"

Takes a list of finger states (1 = open, 0 = closed) and returns the corresponding gesture name.

7. Main Loop
   while True:

Runs continuously until the user quits.

8. Reading and Processing Frame
   ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(frame_rgb)

Reads a frame from the webcam.
Converts to RGB (Mediapipe requires RGB format).
Processes the frame to detect hands and landmarks.

9. Drawing Landmarks and Detecting Fingers
    if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

Draws dots (landmarks) and lines (connections) on the detected hand.

10. Finger State Detection
    lm_list = []
h, w, _ = frame.shape
for id, lm in enumerate(hand_landmarks.landmark):
    lm_list.append((int(lm.x * w), int(lm.y * h)))

# Thumb
fingers.append(1 if lm_list[finger_tips[0]][0] > lm_list[finger_tips[0] - 1][0] else 0)

# Other fingers
for tip in finger_tips[1:]:
    fingers.append(1 if lm_list[tip][1] < lm_list[tip - 2][1] else 0)

Converts normalized coordinates to pixel coordinates.
For the thumb, checks x-axis position.
For other fingers, checks y-axis position.

11. Display Gesture Name
    gesture = detect_gesture(fingers)
cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

Gets the gesture name and displays it on the frame.

12. Show Webcam Output
    cv2.imshow("Gesture Detection", frame)

Shows the camera feed with hand landmarks and gesture text.

13. Exit Condition
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
    break

Press the Escape key to quit.

14. Cleanup
    cap.release()
cv2.destroyAllWindows()

Releases the webcam and closes OpenCV windows.



# ✋ Real-Time Gesture Detection using Mediapipe & OpenCV

This Python project detects hand gestures in real time using Google's Mediapipe Hand Tracking solution. It identifies whether your hand is showing a **Fist**, **Thumbs Up**, **Pointing**, or an **Open Hand**, and displays the gesture name on the screen.

---

## 🚀 Features
- Real-time hand tracking using Mediapipe.
- Recognizes:
  - ✊ Fist
  - 👍 Thumbs Up
  - 👆 Pointing
  - 🖐 Open Hand
- Draws hand landmarks & connections.
- Works on most webcams without GPU.

---

## 🛠 Technologies Used
- **Python 3**
- **OpenCV** – For image processing & display.
- **Mediapipe** – For hand detection and landmark tracking.

---

## 📌 How It Works
1. Mediapipe detects hand landmarks (21 points).
2. Specific landmark positions determine if fingers are open or closed.
3. Based on finger states, the gesture is classified.
4. The gesture name is displayed in real time.

---

## ▶️ Installation & Usage
```bash
# Install dependencies
pip install opencv-python mediapipe
# Run the program
python gesture_detection.py

Ensure your webcam is connected.
Press Esc to quit.

Adjustable Parameters
max_num_hands=1
min_detection_confidence=0.7
min_tracking_confidence=0.7

Increase max_num_hands to track multiple hands.
Adjust detection & tracking confidence for accuracy.

# Example Output
When you make a gesture:
Landmarks & connections appear on your hand.
Gesture name appears on top of the video feed.
