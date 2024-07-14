import cv2
import mediapipe
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
from skimage.feature import hog
from statistics import mean


video = cv2.VideoCapture("PD/readText/001PD_S10_readText.webm")
framerate = video.get(cv2.CAP_PROP_FPS)
options = FaceDetectorOptions(base_options=BaseOptions(model_asset_path='detector.tflite'), running_mode=RunningMode.VIDEO)
bound_box_sizes = ([], [])
cropped_frames = []

with FaceDetector.create_from_options(options) as detector:
    read_success, frame_count = True, -1
    while read_success and video.isOpened():
        read_success, frame = video.read()
        frame_count += 1
        if not read_success: continue
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect_for_video(mp_image, int(1000 * frame_count / framerate))
        assert len(detection_result.detections) == 1, f"{len(detection_result.detections)} faces detected (instead of 1)."
        bbox = detection_result.detections[0].bounding_box
        bound_box_sizes[0].append(bbox.width)
        bound_box_sizes[1].append(bbox.height)
        cropped_frames.append(frame[bbox.origin_y: bbox.origin_y + bbox.height, bbox.origin_x: bbox.origin_x + bbox.width].copy())
        cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (255, 0, 0), 3)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()

avg_width, avg_height = int(mean(bound_box_sizes[0])), int(mean(bound_box_sizes[1]))
for ind in range(len(cropped_frames)):
    cropped_frames[ind] = cv2.resize(cropped_frames[ind], (avg_width, avg_height))

for ind in range(len(cropped_frames)):
    hog_feats = hog(cv2.cvtColor(cropped_frames[ind], cv2.COLOR_BGR2GRAY))
    cv2.imshow("cropped frame", cropped_frames[ind])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
