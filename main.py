import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import cv2 as cv


videopath = "/home/danih/ETUDES/5A/TAI/input/video.mp4"
modelpath = "/home/danih/ETUDES/5A/TAI/Projet3_AnalyseSportive/models/pose_landmarker_full.task"

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image




BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=modelpath),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=2
    )

with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    cap = cv.VideoCapture(videopath)
    fps = cap.get(cv.CAP_PROP_FPS)
    numero_frame = 0
    while True:
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        timestamp_ms = int(numero_frame / fps * 1000)

        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated_image = draw_landmarks_on_image(rgb_image, pose_landmarker_result)

        cv.imshow('frame', cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) == ord('q'):
            break
        numero_frame += 1
    
    cap.release()
    cv.destroyAllWindows()