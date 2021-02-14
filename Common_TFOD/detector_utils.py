# Utilities for object detector.

import numpy as np
import sys
from math import pow, sqrt
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
from playsound import playsound
from scipy.spatial import distance
import cv2
from Common_TFOD import label_map_util
from collections import defaultdict
# from Common_TFOD import alertcheck
detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'
Focal =515
NUM_CLASSES = 2
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a=b=0

# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def get_points_frame(num_face_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np,Orientation):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a,b
    # global point1,point2,distace,ID
    ID = []
    point1= []
    point2 = []
    distance =[]
    hand_cnt=0
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    for i in range(num_face_detect):
        
        if (scores[i] > score_thresh):

            if classes[i] == 1: 
                id = 'Masked'
                #b=1
            if classes[i] == 2:
                id ='Without_Mask'
                avg_width = 3.0 # To compensate bbox size change
                #b=1
            
            # if i == 0: color = color0
            # else: color = color1
            color = color1
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            dist = distance_to_camera(avg_width, focalLength, int(right-left))
            point1.append(p1)
            point2.append(p2)
            distance.append(dist)
            ID.append(id)

            
    return point1,point2,ID #,distance

def draw_box(img,coll,close,num_face_detect,scores,ID):

    for i in range(num_face_detect):
        if i in close:
            p1,p2 =coll[i]
            color = (255, 0, 0)
            cv2.rectangle(img, p1, p2, color, 3, 1)

            cv2.putText(img, 'Object ' + str(i) + ': ' + ID[i], (int(p1[0]), int(p1[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(img, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(p1[0]), int(p1[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # cv2.putText(image_np, 'distance from camera: ' + str("{0:.2f}".format(dist) + ' inches'),
            #             # (int(im_width*0.65),int(im_height*0.9+30*i)),
            #             (int(left), int(top) - 35),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

            cv2.putText(img, 'Number of Object: ' + str("{0:.2f}".format(num_face_detect)),
                        (int(30), int(460)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
            posii = int(img.shape[1] / 2)
            cv2.putText(img, "ALERT", (posii, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            playsound(r"D:\Virtual_Env\GitHub_projects\Face Mask Detection\Common_TFOD\alert.wav")
        else:


            p1, p2 = coll[i]
            color = (220,50,255)
            cv2.rectangle(img, p1, p2, color, 3, 1)

            cv2.putText(img, 'Object ' + str(i) + ': ' + ID[i], (int(p1[0]), int(p1[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(img, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(p1[0]), int(p1[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # cv2.putText(image_np, 'distance from camera: ' + str("{0:.2f}".format(dist) + ' inches'),
            #             # (int(im_width*0.65),int(im_height*0.9+30*i)),
            #             (int(left), int(top) - 35),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

            cv2.putText(img, 'Number of Object: ' + str("{0:.2f}".format(num_face_detect)),
                        (int(30), int(460)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes),np.squeeze(num)


def get_mid_point(p1,p2,Number_Detect):
    mid = []
    coll = []
    height= []
    for i in range(Number_Detect):
        midx = round((p1[i][0]+p2[i][0])/2,4)
        midy = round((p1[i][1]+p2[i][1])/2,4)
        mid1 = (midx,midy)
        mid.append(mid1)
        h = round(p2[i][1]-p1[i][1],4)
        height.append(h)
        coll.append([p1[i],p2[i]])
    return mid,coll,height
def get_distance(midpoints,height,num_detect):
    dist = dict()

    # dist= np.zeros(num_detect,num_detect)
    # dist = np.zeros((num_detect,num_detect))
    for i in range(num_detect):
        # distance = (165 * 515)/height[i]
        # midx_cm = (midpoints[i][0]*distance)/515
        # midy_cm = (midpoints[i][1]*distance)/515
        distance = (55 * Focal) / height[i]
        midx_cm = (midpoints[i][0] * distance) / Focal
        midy_cm = (midpoints[i][1] * distance) / Focal
        dist[i] = (midx_cm,midy_cm,distance)
    #     for j in range(i+1,num_detect):
    #         if i!=j:
    #             dst = distance.euclidean(midpoints[i],midpoints[j])
    #             dist[i][j]=dst
    return dist
#
def get_closest(dist,num,thresh):
    p1= set()
    # p2=[]
    # d=[]
    for i in dist.keys():#range(num):
        for j in dist.keys():#range(i,num):
            # if( (i!=j) & (dist[i][j]<=thresh)):
            if i<j:
                d = sqrt(pow(dist[i][0] - dist[j][0], 2) + pow(dist[i][1] - dist[j][1], 2) + pow(
                    dist[i][2] - dist[j][2], 2))
                if d<150:
                    p1.add(i)
                    p1.add(j)
                # d.append(dist[i][j])
    return p1#,p2,d
