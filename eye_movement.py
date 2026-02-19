import cv2
import dlib
import numpy as np

# Load dlib’s face detector and 68 landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\butta\FYP - 1\Cheating-Surveillance-System\models\shape_predictor_68_face_landmarks.dat")

def detect_pupil(eye_region):
    if eye_region.size == 0:
        return None, None
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, threshold_eye = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        pupil_contour = max(contours, key=cv2.contourArea)
        px, py, pw, ph = cv2.boundingRect(pupil_contour)
        return (px + pw // 2, py + ph // 2), (px, py, pw, ph)
    return None, None

def process_eye_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_direction = "Looking Center"

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract left and right eye landmarks
        left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        
        # Get bounding rectangles for the eyes
        left_eye_rect = cv2.boundingRect(left_eye_points)
        right_eye_rect = cv2.boundingRect(right_eye_points)
        
        # Extract eye regions safely
        l_x, l_y, l_w, l_h = left_eye_rect
        r_x, r_y, r_w, r_h = right_eye_rect

        left_eye = frame[l_y:l_y + l_h, l_x:l_x + l_w] if l_h > 0 and l_w > 0 else np.zeros((1,1,3), np.uint8)
        right_eye = frame[r_y:r_y + r_h, r_x:r_x + r_w] if r_h > 0 and r_w > 0 else np.zeros((1,1,3), np.uint8)
        
        # Detect pupils
        left_pupil, _ = detect_pupil(left_eye)
        right_pupil, _ = detect_pupil(right_eye)
        
        # Draw bounding boxes
        cv2.rectangle(frame, (l_x, l_y), (l_x + l_w, l_y + l_h), (0, 255, 0), 2)
        cv2.rectangle(frame, (r_x, r_y), (r_x + r_w, r_y + r_h), (0, 255, 0), 2)
        
        # Draw pupils
        if left_pupil:
            cv2.circle(frame, (l_x + left_pupil[0], l_y + left_pupil[1]), 5, (0, 0, 255), -1)
        if right_pupil:
            cv2.circle(frame, (r_x + right_pupil[0], r_y + right_pupil[1]), 5, (0, 0, 255), -1)
        
        # Robust gaze detection
        if left_pupil and right_pupil:
            # Normalize pupil position relative to each eye
            lx_norm = left_pupil[0] / max(l_w, 1)
            rx_norm = right_pupil[0] / max(r_w, 1)
            ly_norm = (left_pupil[1] / max(l_h,1) + right_pupil[1] / max(r_h,1)) / 2

            if lx_norm < 0.35 and rx_norm < 0.35:
                gaze_direction = "Looking Left"
            elif lx_norm > 0.65 and rx_norm > 0.65:
                gaze_direction = "Looking Right"
            elif ly_norm < 0.4:
                gaze_direction = "Looking Up"
            elif ly_norm > 0.6:
                gaze_direction = "Looking Down"
            else:
                gaze_direction = "Looking Center"
    
    return frame, gaze_direction
