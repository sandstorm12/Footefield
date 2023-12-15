import cv2
import mediapipe as mp
import numpy as np

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_5/color/color00000.jpg")
annotated_image = np.copy(image.numpy_view())

print(annotated_image.shape)

cv2.imshow("frame", annotated_image)
cv2.waitKey(0)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)
pose_landmarks_list = detection_result.pose_landmarks
print(len(pose_landmarks_list), len(pose_landmarks_list[0]))

for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())

cv2.imshow("frame", annotated_image)
cv2.waitKey(0)
