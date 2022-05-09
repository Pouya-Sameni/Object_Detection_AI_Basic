from ntpath import join
from imageai.Detection import ObjectDetection
import os

executionPath = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()

detector.setModelPath(os.path.join(executionPath, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(
    
    input_image=os.path.join(executionPath,"test2.jpg"),
    output_image_path=os.path.join(executionPath, "RefactoredImaged.jpg")
    
    )

for item in detections:
    print (item["name"],
    " : ",
    item["percentage_probability"])



