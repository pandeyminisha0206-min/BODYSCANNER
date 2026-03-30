import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

scan_y = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = result.pose_landmarks.landmark

        # Head position
        head_x = int(landmarks[0].x * w)
        head_y = int(landmarks[0].y * h)

        # Chest
        chest_x = int(landmarks[12].x * w)
        chest_y = int(landmarks[12].y * h)

        # Stomach
        stomach_x = int(landmarks[24].x * w)
        stomach_y = int(landmarks[24].y * h)

        # Draw highlight circles
        cv2.circle(frame,(head_x,head_y),25,(255,0,255),2)
        cv2.circle(frame,(chest_x,chest_y),30,(0,255,0),2)
        cv2.circle(frame,(stomach_x,stomach_y),30,(0,255,255),2)

        cv2.putText(frame,"AI ANALYZING BODY",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),3)

    # scanning laser line
    scan_y += 5
    if scan_y > h:
        scan_y = 0

    cv2.line(frame,(0,scan_y),(w,scan_y),(0,255,255),2)

    cv2.imshow("AI Health Scanner",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()