import streamlit as st
from PIL import Image
import cv2
import numpy as np
import darknet

def main():
    selected_box = st.sidebar.selectbox('Choose one of the following', ('Welcome', 'CLAHE', 'Object Detection'))

    if selected_box == 'Welcome':
        welcome()
    elif selected_box == 'CLAHE':
        photo()
    elif selected_box == 'Object Detection':
        object_detection()


def welcome():
    st.title('CLAHE and Object Detection using Streamlit')


def photo():
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        image = original_image.rotate(90)

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>', unsafe_allow_html=True)
            st.image(image, width=400)

        with col2:
            st.markdown('<p style="text-align: center;">After</p>', unsafe_allow_html=True)
            filter = st.sidebar.radio('Covert your photo to:', ['Original', 'CLAHE colored', 'CLAHE b&w'])

            if filter == 'CLAHE colored':
                converted_img = np.array(image.convert('RGB'))
                lab_img = cv2.cvtColor(converted_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab_img)
                clahe = cv2.createCLAHE(clipLimit=5)
                clahe_img = clahe.apply(l)
                updated_lab_img2 = cv2.merge((clahe_img, a, b))
                CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

                st.image(CLAHE_img, width=400)

            elif filter == 'CLAHE b&w':
                converted_img = np.array(image.convert('RGB'))
                image_bw = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=7)
                final_img = clahe.apply(image_bw)
                st.image(final_img, width=400)

            else:
                st.image(image, width=400)


def object_detection():
    uploaded_file = st.file_uploader("Upload an image for object detection", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        image = original_image.rotate(90)

        st.image(image, width=600)

        model_names = ['YOLO V4', 'YOLO V7', 'SSD (Single Shot Detector)', 'Faster R-CNN', 'RetinaNet']
        model_selection = st.sidebar.selectbox('Choose an Object Detection Model', model_names)
        confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)

        if model_selection == 'YOLO V4':
            object_detection_results = perform_object_detection_yolo_v4(image, confidence_threshold)
        elif model_selection == 'YOLO V7':
            object_detection_results = perform_object_detection_yolo_v7(image, confidence_threshold)
        elif model_selection == 'SSD (Single Shot Detector)':
            object_detection_results = perform_object_detection_ssd(image, confidence_threshold)
        elif model_selection == 'Faster R-CNN':
            object_detection_results = perform_object_detection_faster_rcnn(image, confidence_threshold)
        elif model_selection == 'RetinaNet':
            object_detection_results = perform_object_detection_retinanet(image, confidence_threshold)
        else:
            object_detection_results = []

        st.subheader('Object Detection Results')
        for result in object_detection_results:
            label = result['label']
            confidence = result['confidence']
            bounding_box = result['bounding_box']

            st.write(f'Label: {label}, Confidence: {confidence}')
            st.image(draw_bounding_box(image, bounding_box), width=400)

        st.markdown("---")


def perform_object_detection_yolo_v4(image):
    # Load YOLO V4 model
    net = darknet.load_net("path/to/yolov4.cfg", "path/to/yolov4.weights", 0)
    meta = darknet.load_meta("path/to/coco.data")

    # Prepare image for detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (darknet.network_width(net), darknet.network_height(net)), interpolation=cv2.INTER_LINEAR)
    image_darknet = darknet.make_image(darknet.network_width(net), darknet.network_height(net), 3)
    darknet.copy_image_from_bytes(image_darknet, image_resized.tobytes())

    # Perform object detection
    detections = darknet.detect_image(net, meta, image_darknet, thresh=0.5)

    # Process detection results
    object_detection_results = []
    for detection in detections:
        label = detection[0].decode()
        confidence = detection[1]
        x, y, w, h = detection[2]

        bounding_box = [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]

        object_detection_results.append({'label': label, 'confidence': confidence, 'bounding_box': bounding_box})

    return object_detection_results


def perform_object_detection_yolo_v7(image):
    # Load the YOLO V7 model and perform object detection
    net = cv2.dnn.readNetFromDarknet("path/to/yolov7.cfg", "path/to/yolov7.weights") #replace 
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Define class names and confidence threshold
    class_names = ["class1", "class2", "class3", ...]  # Replace with actual class names
    confidence_threshold = 0.5  # Set an appropriate confidence threshold

    # Get the image dimensions
    image_height, image_width = image.shape[:2]

    # Perform object detection
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Process the output detections
    object_detection_results = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                label = class_names[class_id]
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)
                x = int(center_x - width/2)
                y = int(center_y - height/2)

                object_detection_results.append({
                    'label': label,
                    'confidence': confidence,
                    'bounding_box': [x, y, x + width, y + height]
                })

    return object_detection_results

def perform_object_detection_ssd(image, class_names, confidence_threshold=0.5):
    # Load the pre-trained SSD model
    model_weights = "path/to/ssd_weights.pb"  # Replace with the path to the SSD model weights
    model_config = "path/to/ssd_config.pbtxt"  # Replace with the path to the SSD model configuration file

    net = cv2.dnn.readNetFromTensorflow(model_weights, model_config)

    # Preprocess the image for object detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (127.5, 127.5, 127.5), True, False)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass through the network to get the detections
    detections = net.forward()

    # Process the detections
    object_detection_results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            label = class_names[class_id]
            bounding_box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            x, y, w, h = bounding_box.astype(np.int)
            object_detection_results.append({
                'label': label,
                'confidence': confidence,
                'bounding_box': [x, y, x + w, y + h]
            })

    return object_detection_results


def perform_object_detection_faster_rcnn(image):
    # Load the pre-trained Faster R-CNN model
    model_weights = "path/to/faster_rcnn_weights.caffemodel"  # Replace with the path to the Faster R-CNN model weights
    model_config = "path/to/faster_rcnn_config.prototxt"  # Replace with the path to the Faster R-CNN model configuration file

    net = cv2.dnn.readNet(model_weights, model_config)

    # Preprocess the image for object detection
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass through the network to get the detections
    detections = net.forward()

    # Process the detections
    object_detection_results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust the confidence threshold as needed
            class_id = int(detections[0, 0, i, 1])
            label = str(class_id)
            bounding_box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            x, y, w, h = bounding_box.astype(np.int)
            object_detection_results.append({
                'label': label,
                'confidence': confidence,
                'bounding_box': [x, y, x + w, y + h]
            })

    return object_detection_results


def perform_object_detection_retinanet(image):
    # Load the pre-trained RetinaNet model
    model_weights = "path/to/retinanet_weights.caffemodel"  # Replace with the path to the RetinaNet model weights
    model_config = "path/to/retinanet_config.prototxt"  # Replace with the path to the RetinaNet model configuration file

    net = cv2.dnn.readNet(model_weights, model_config)

    # Preprocess the image for object detection
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass through the network to get the detections
    detections = net.forward()

    # Process the detections
    object_detection_results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust the confidence threshold as needed
            class_id = int(detections[0, 0, i, 1])
            label = str(class_id)
            bounding_box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            x, y, w, h = bounding_box.astype(np.int)
            object_detection_results.append({
                'label': label,
                'confidence': confidence,
                'bounding_box': [x, y, x + w, y + h]
            })

    return object_detection_results

def draw_bounding_box(image, bounding_box):
    # Helper function to draw bounding box on the image
    img_with_bbox = image.copy()
    x1, y1, x2, y2 = bounding_box
    cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_with_bbox



if __name__ == "__main__":
    main()
