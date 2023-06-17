import json
import numpy as np
import onnxruntime as ort
import cv2
import torch
from glob import glob

class Yolov8:
    def __init__(self, onnx_model, conf_thresh, nms_thresh) -> None:
        self.onnx_model = onnx_model
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.classes = {0:'person', 1:'head', 2:'door_open', 3:'door_half_open', 4:'door_close'}
        self.input_height = 640
        self.input_width = 640

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self, input_image):
        self.input_image = input_image

        self.img = cv2.imread(self.input_image)
        self.img_height, self.img_width = self.img.shape[:2]

        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.input_width, self.input_height))

        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1)) # h, w, c ==> c, h, w
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data
    
    def postprocess(self, output):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []

        x_ratio = self.img_width / self.input_width
        y_ratio = self.img_height / self.input_height

        for i in range(rows):
            class_scores = outputs[i][4:]
            max_score = np.amax(class_scores)

            if max_score > self.conf_thresh:
                class_id = np.argmax(class_scores)
            
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = int((x - w / 2) * x_ratio)
                top = int((y - h / 2) * y_ratio)
                width = int(w * x_ratio)
                height = int(h * y_ratio)

                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        # print("NMS前目标个数: ", len(class_ids))
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.nms_thresh)
        # print("有效目标数：", len(indices))

        object_info = []

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            object_info.append({
                "x" : int(box[0]),
                "y" : int(box[1]),
                "width" : int(box[2]),
                "height" : int(box[3]),
                "confidence" : float(score),
                "name" : self.classes[int(class_id)]
            })
        # Return the modified input image
        return object_info

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def init():
    """
    Initialize model
    Returns: model
    """
    onnx_model = 'models/elevator-close-detv8s.onnx'
    model = Yolov8(onnx_model, 0.75, 0.5)
    return model

def process_image(model, input_image, args=None):
    """
    Do inference to analysis input_image and get output
    Attributes:
    handle: algorithm handle returned by init()
    input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
    Returns: process result
    """
    session = ort.InferenceSession(model.onnx_model, providers=['CPUExecutionProvider'])
    # Process image here
    image_data = model.preprocess(input_image)
    model_input = session.get_inputs()
    outputs = session.run(None, {model_input[0].name: image_data})
    object_infos = model.postprocess(outputs)

    fake_result = {}
    fake_result["algorithm_data"] = {
        "is_alert": False,
        "target_count": 0,
        "target_info": []
    }

    for info in object_infos:
        print(info)
        if info["name"] == "door_open":
            fake_result["algorithm_data"]["is_alert"] = True
            fake_result["algorithm_data"]["target_count"] += 1
            fake_result["algorithm_data"]["target_info"].append(info)

    fake_result["model_data"] = {
        "objects": object_infos
    }

    return json.dumps(fake_result, indent=4)
    


def main():
    image_path = 'assert/ZDSzudangelevator20230320_V3_train_elevator_1_003221.jpg'
    model = init()
    json_content = process_image(model, image_path)
    json_file = open("detectInfo.json", 'w')
    json_file.write(json_content)
    json_file.close()

if __name__ == "__main__":
    main()

