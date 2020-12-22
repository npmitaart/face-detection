# Author: Giodio Mitaart & Yesiana Phann

'''
IMPORT LIBRARIES
'''
# import all the libraries that will be used
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os


'''
GET PATH LIST
'''
# getting the lists of our path that contains the dataset
def get_path_list(root_path):
    for i, name_train in enumerate(os.listdir(root_path)):
        name_train_path = root_path+name_train
        return name_train


'''
GET CLASS ID
'''
# getting the class id (all the images will be labeled using this function)
def get_class_id(root_path, train_names):
    img_train = []
    img_class = []

    #looping every data in our path and return them as labeled data
    for i, name_train in enumerate(os.listdir(root_path)):
        name_train_path = root_path+name_train
        for name in os.listdir(name_train_path):
            img = name_train_path + '/' + name
            img_train.append(img)
            img_class.append(i)
    return img_train, img_class


'''
DETECT TRAIN FACES AND FILTER
'''
# train the face detection
def detect_train_faces_and_filter(image_list, image_classes_list):
    train_face_grays = []
    filtered_classes_list = image_classes_list

    # init cascade function to trained images both positive and negative
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for image in image_list:
        img_gray = cv2.imread(image,0)
        # doing the detection
        detected_face = cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
        if(len(detected_face)<1):
            continue
        for face_rect in detected_face:
            x,y,h,w = face_rect
            # cropping the image to fit the face
            face_img = img_gray[y:y+h, x:x+w] 
            train_face_grays.append(face_img)
    return train_face_grays, filtered_classes_list


'''
TRAIN
'''
# preparing the images in order to recognize every classes image we have
def train(train_face_grays, image_classes_list):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))
    return recognizer


'''
GET TEST IMAGE DATA
'''
# getting the images to be tested
def get_test_images_data(test_root_path):
    img_list = []
    for name_test in os.listdir('dataset/test'):
        img_bgr= test_root_path+name_test
        img_list.append(img_bgr)
    return img_list


'''
DETECT TEST FACES AND FILTER
'''
# doing the testing of all image lists
def detect_test_faces_and_filter(image_list):
    test_faces_gray = []
    test_faces_rect = []
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for image in image_list:
        img_bgr = cv2.imread(image)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # perform the cascade detection
        detected_face = cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
        
        if(len(detected_face)<1):
            continue
        for face_rect in detected_face:
            x,y,h,w = face_rect
            face_img = img_gray[y:y+h, x:x+w]
            test_faces_gray.append(face_img)
            test_faces_rect.append(face_rect)

    return test_faces_gray, test_faces_rect


'''
PREDICT
'''
# make a prediction of every result of the test process before
def predict(recognizer, test_image_list):
    predict_result = []
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for image in test_image_list:
        # detecting every faces in test_image_list
        detected_face = cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
        
        if(len(detected_face)<1):
            continue
        for face_rect in detected_face:
            # perform the recognizer to predict the image
            predict = recognizer.predict(image)
            predict_result.append(predict)
    return predict_result


'''
DRAW PREDICTION RESULTS
'''
# make a function to get all the result and the percentage of face detection
def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, size):
    final_rects = []
    name = os.listdir("dataset/train")
    # looping every prediction
    for face_rect,image,pred_result in zip(test_faces_rects,test_image_list,predict_results):
        x,y,h,w = face_rect
        res,conf = pred_result
        img_bgr= cv2.imread(image)
        # calculate the percentage of face detection
        confidence = math.floor(conf *100)/100
        txt = name[res]+' '+str(confidence)+"%"
        cv2.putText(img_bgr, txt, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0) , 3)
        # rectangle
        cv2.rectangle(img_bgr, (x,y), (x+w, y+h), (0,255,0))
        final_rects.append(img_bgr)
    return final_rects


'''
COMBINE AND SHOW RESULT
'''
# combine all the result and percentage and show them as the output
def combine_and_show_result(image_list, size):
    final = []
    for image in image_list:
        images = cv2.resize(image,(size,size))
        final.append(images)
    concated_img = cv2.hconcat(final)
    cv2.imshow('result',concated_img)
    cv2.waitKey(0)

'''
MAIN TO CALL ALL THE FUNCTION ABOVE
'''
if __name__ == "__main__":
    
    train_root_path = 'dataset/train/'

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, filtered_classes_list = detect_train_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    test_root_path =  'dataset/test/'

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, 200)
    
    combine_and_show_result(predicted_test_image_list, 200)