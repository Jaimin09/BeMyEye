import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from gtts import gTTS 
import os
import speech_recognition as sr 

# Importing OpenCV to capture video from webcam
import cv2
cap = cv2.VideoCapture(0)

sys.path.append("models/research/")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
PATH_TO_LABELS = os.path.join('models', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

obj_to_width = {
    'bottle' : 7.5,
    'cup' : 10,
    'chair' : 58,
    'apple' : 8,
    'orange' : 8,
    'tv' : 102,
    'mouse' : 6.5,
    'laptop' : 37.5,
    'remote' : 6,
    'cell phone' : 7,
    'book' : 13
}

id_to_obj = {}
for i in range(len(categories)):
    id_to_obj[categories[i]['id']] = categories[i]['name'];
    
def find_distance(known_width, focal_length, pixel_width):
    return (known_width*focal_length)/pixel_width

def text_to_speech(text):
    speech = gTTS(text = text, lang = 'en', slow = False)
    speech.save('text.mp3')
    os.system('start text.mp3')
    os.remove('text.mp3')

def write_distances_on_image(image, boxes, scores, classes, id_to_obj, obj_to_width, focal_length = 1035):
    num_obj = sum(sum(scores > 0.5))
    global ob
    text=ob+" not found"
    for i in range(num_obj):
        x1 = int(boxes[0][i][1]*800)
        y1 = int(boxes[0][i][0]*600)
        x2 = x1 + 250
        y2 = y1 + 30
        
        ## Block to find the distance
        px_width = (boxes[0][i][3] - boxes[0][i][1])*800
        px_height = (boxes[0][i][2] - boxes[0][i][0])*600
        
        pxl = min(px_width, px_height)
        
        obj = id_to_obj[classes[0][i]]
        try:
            rl_width = obj_to_width[obj]
            dis = int(find_distance(rl_width, focal_length, pxl))
                
        except KeyError:
            dis = 0
        ## Block Ends here    
        if(obj==ob):
            text=ob+" found at a distance of "+str(dis)
            text_to_speech(text)
            ob=""
        print(text)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color = (255, 255, 255), thickness = -1)
        image = cv2.putText(image, "dis : " + str(dis) + " cms", (x1+15, y1+20), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = (0,0,0), thickness = 2)
        
    return image
r = sr.Recognizer()
ob=""
with sr.Microphone() as source:
    print("Talk")
    audio_text = r.listen(source)
    print("Time over, thanks")
    try:
        # using google speech recognition
        text=r.recognize_google(audio_text)
        #print("Text: "+text)
        t=text.split(' ')
        for x in t:
            if(x!='locate' and x!='find' and x!='me' and x!='a' and x!='an' and x!='of'):
                if(len(ob)!=0):
                    ob=ob+" "+x
                else:
                    ob=x
        ob.strip()
        print(ob)
    except:
         text_to_speech("Sorry, I did not get that")
cap = cv2.VideoCapture(0)
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while len(ob)!=0:
            ret, image_np = cap.read()
            #ob="cell phone"
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
        
            image_np = write_distances_on_image(image_np, boxes, scores, classes, id_to_obj, obj_to_width, focal_length = 1035)
        
            cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                break
                


"""while True:
    ret, image_np = cap.read()
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
    
    cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
    if cv2.waitKey(5) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break"""