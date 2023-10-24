import numpy as np
import cv2
from time import time
import sys

import json
import onnxruntime
import glob
import numpy as np
from PIL import Image


class ObjectDetection:
    """
    The class performs generic object detection on a video file.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot boxes on objects along with labels.
    Upcoming Features:
    """

    def __init__(self, input_file, out_file="LMD_Video.avi"):
        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """

        labels_file = "labels.json"
        with open(labels_file) as f:
            self.classes = json.load(f)

        self.input_file = input_file
        self.session = self.load_fasterrcnn_model()
        self.out_file = out_file
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_from_file(self):
        """
        Function creates a streaming object to read the video from the file frame by frame.
        :param self:  class object
        :return:  OpenCV object to stream video frame by frame.
        """
        cap = cv2.VideoCapture(self.input_file)
        assert cap is not None
        return cap
    
    def load_fasterrcnn_model(self):

        try:
            session = onnxruntime.InferenceSession("model.onnx")
            print("ONNX model loaded...")
        except Exception as e: 
            print("Error loading ONNX file: ", str(e))

        return session
    
    def preprocess(self, image, height_onnx, width_onnx):
        """Perform pre-processing on raw input image
        
        :param image: raw input image
        :type image: PIL image
        :param height_onnx: expected height of an input image in onnx model
        :type height_onnx: Int
        :param width_onnx: expected width of an input image in onnx model
        :type width_onnx: Int
        :return: pre-processed image in numpy format
        :rtype: ndarray 1xCxHxW
        """

        image = image.convert('RGB')
        image = image.resize((width_onnx, height_onnx))
        np_image = np.array(image)
        # HWC -> CHW
        np_image = np_image.transpose(2, 0, 1) # CxHxW
        # normalize the image
        mean_vec = np.array([0.485, 0.456, 0.406])
        std_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(np_image.shape).astype('float32')
        for i in range(np_image.shape[0]):
            norm_img_data[i,:,:] = (np_image[i,:,:] / 255 - mean_vec[i]) / std_vec[i]
        np_image = np.expand_dims(norm_img_data, axis=0) # 1xCxHxW
        return np_image

    def get_predictions_from_ONNX(self, onnx_session, img_data):
        """perform predictions with ONNX runtime
        
        :param onnx_session: onnx model session
        :type onnx_session: class InferenceSession
        :param img_data: pre-processed numpy image
        :type img_data: ndarray with shape 1xCxHxW
        :return: boxes, labels , scores 
                (No. of boxes, 4) (No. of boxes,) (No. of boxes,)
        :rtype: tuple
        """

        sess_input = onnx_session.get_inputs()
        sess_output = onnx_session.get_outputs()
        
        # predict with ONNX Runtime
        output_names = [output.name for output in sess_output]
        predictions = onnx_session.run(output_names=output_names,\
                                                input_feed={sess_input[0].name: img_data})

        return output_names, predictions


    def _get_box_dims(self, image_shape, box):
        box_keys = ['topX', 'topY', 'bottomX', 'bottomY']
        height, width = image_shape[0], image_shape[1]

        box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

        box_dims['topX'] = box_dims['topX'] * 1.0 / width
        box_dims['bottomX'] = box_dims['bottomX'] * 1.0 / width
        box_dims['topY'] = box_dims['topY'] * 1.0 / height
        box_dims['bottomY'] = box_dims['bottomY'] * 1.0 / height

        return box_dims

    def _get_prediction(self, boxes, labels, scores, image_shape, classes):
        bounding_boxes = []
        for box, label_index, score in zip(boxes, labels, scores):
            box_dims = self._get_box_dims(image_shape, box)

            box_record = {'box': box_dims,
                        'label': classes[label_index],
                        'score': score.item()}

            bounding_boxes.append(box_record)

        return bounding_boxes

    def score_frame(self, frame):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.
        """

        batch_size = 1
        batch, channel, height_onnx, width_onnx = self.session.get_inputs()[0].shape
        print(height_onnx)
        print(width_onnx)
        cv2.imwrite("tmp.jpg", frame)
        tmp_img = Image.open("tmp.jpg")
        img_data = self.preprocess(tmp_img, height_onnx, width_onnx)
        output_names, predictions = self.get_predictions_from_ONNX(self.session, img_data)

        # Filter the results with threshold.
        # Please replace the threshold for your test scenario.
        score_threshold = 0.8

        # in case of retinanet change the order of boxes, labels, scores to boxes, scores, labels
        # confirm the same from order of boxes, labels, scores output_names 
        boxes, labels, scores = predictions[0], predictions[1], predictions[2]
        bounding_boxes = self._get_prediction(boxes, labels, scores, (height_onnx, width_onnx), self.classes)
        filtered_bounding_boxes = [box for box in bounding_boxes if box['score'] >= score_threshold]
        
        return filtered_bounding_boxes

    def plot_boxes(self, results, frame):
        """
        plots boxes and labels on frame.
        :param results: inferences made by model
        :param frame: frame on which to  make the plots
        :return: new frame with boxes and labels plotted.
        """
        image_boxes = results # replace with desired image index
        y, x, channels = frame.shape

        print(frame.shape)

        # Draw box and label for each detection 
        for detect in image_boxes:
            label = detect['label']
            box = detect['box']
            ymin, xmin, ymax, xmax =  box['topY'], box['topX'], box['bottomY'], box['bottomX']
            topleft_x, topleft_y = x * xmin, y * ymin
            botleft_x, botleft_y = x * xmax, y * ymax
            width, height = x * (xmax - xmin), y * (ymax - ymin)
            print('{}: {}, {}, {}, {}'.format(detect['label'], topleft_x, topleft_y, width, height))
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (int(topleft_x), int(topleft_y)), (int(botleft_x), int(botleft_y)), bgr, 2)
            cv2.putText(frame, label, (int(topleft_x), int(topleft_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        player = self.get_video_from_file() # create streaming service for application
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        fc = 0
        fps = 0
        tfc = int(player.get(cv2.CAP_PROP_FRAME_COUNT))
        tfcc = 0
        while True:
            fc += 1
            start_time = time()
            ret, frame = player.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps += 1/np.round(end_time - start_time, 3)
            if fc == 10:
                fps = int(fps / 10)
                tfcc += fc
                fc = 0
                per_com = int(tfcc / tfc * 100)
                print(f"Frames Per Second : {fps} || Percentage Parsed : {per_com}")
            out.write(frame)
        player.release()


link = sys.argv[1]
output_file = sys.argv[2]
a = ObjectDetection(link, output_file)
a()