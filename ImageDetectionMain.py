from ntpath import join
from imageai.Detection import ObjectDetection
from imageai.Detection.Custom import CustomObjectDetection
import os
import warnings

#suppress warnings
warnings.filterwarnings('ignore')


executionPath = os.getcwd()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()

detector.setModelPath("FaceDetectionModel V2.h5")

detector.setJsonPath("detection_config.json")

detector.loadModel()


detections = detector.detectObjectsFromImage(input_image="pouya2.jpg", output_image_path="pouya2-detected.jpg")


for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])