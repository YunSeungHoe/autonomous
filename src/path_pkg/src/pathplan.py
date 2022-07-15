#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import signal
import sys
import os
import random
from path_pkg.msg import path, locate


bridge = CvBridge()
cv_image = np.empty(shape=[0])
def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data,"bgr8")
rospy.init_node('cam_tune', anonymous=True)
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
pub = rospy.Publisher('path', locate, queue_size=10)
out = locate()
Width = 640
Height = 480

CAM_FPS = 30    
WIDTH, HEIGHT = 640, 480    

Width = 640
Height = 480
ROI_ROW = 250   
ROI_HEIGHT = HEIGHT - ROI_ROW   
L_ROW = ROI_HEIGHT - 120  
# Looking for Optimal ROI
# src_point = np.array([[-96, 480], [124, 340], [610, 340], [866, 480]])
#src_point = np.array([[1, 418], [124, 340], [610, 340], [733, 418]]) # based on hoffman
#src_point = np.array([[40,380], [180,300], [520,300], [640, 380]]) 
src_point = np.array([[10,400], [160,330], [500,330], [630, 400]]) 
#src_point = np.array([[20,400], [180,300], [520,300], [670, 400]]) #for optimal
#dst_point = np.array([[50,480], [50,4], [536,4], [536,480]]) # "50" is safe boundary
dst_point = np.array([[50,480], [50,4], [590,4], [590,480]])
src = np.float32(src_point)
dst = np.float32(dst_point)
Mat = cv2.getPerspectiveTransform(src, dst)
Mat_rev = cv2.getPerspectiveTransform(dst, src)
gaussian_th = 3


#canny_low = 170
#canny_high = 120

canny_low = 50
canny_high = 100

win_num = 15
margin = 15


def getBirdView(img, Mat):
    return cv2.warpPerspective(img, Mat, (Width, Height))

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (gaussian_th, gaussian_th), 0)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def lane_candi(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    canny_img = cv2.Canny(v, canny_low, canny_high)
    #cv2.imshow('v channel',v)
    bird_img = getBirdView(canny_img, Mat)
    cv2.imshow('bird_img', bird_img)

    return bird_img

def onMouse(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN :
        print('---- x / y = : ', x, y) # for pixel 
        print("blue, green, red : ",param[y,x]) #for BGR
        print("...")


LEFT = 0 
MID = 1 
RIGHT = 2 
EMPTY = 3 #

#
win_num = 15 # 
margin = 30 #s
min_win = win_num -1 - 5#
minpix = 10 #
minpix_first = 3 #

fit_tmp = [0,0,0]
fit_true = MID
fit_first = False


#============================================= 
# 직선에서 path 생성
# 직선에서는 sliding window를 통해 중앙차선의 방정식을 얻는다.
# 받은 이미지의 y축을 동일한 크기로 나누어서 방정식에 대입
# x, y 좌표를 얻는다.
#============================================= 
def findPointStraight(image, coefA, coefB, coefC, num):
    pointArr = []
    for i in range(0, num):
        valY = image.shape[0] * (i + 1) / (num+1)
        valX = coefA * valY * valY + coefB * valY + coefC 
        pointArr.append([valX, valY])
    return pointArr

#============================================= 
# 곡선에서 path 생성할 때 가정
# 1. 한쪽 차선만 보인다. 
# 2. 보이는 차선은 우회전시 왼쪽차선, 좌회전시 오른쪽차선 이다.
# 차선의 함수가 입력값으로 (2차함수)
# 버드아이뷰에서 x, y 픽셀의 비율이 1:1일 경우
# linePix 차선 폭의 픽셀 수, num은 점의 수
#============================================= 
def findPointCurve(image, coefA, coefB, coefC, linePix, num):
    # linePix = 차선의 픽셀을 의미, 즉 실제 길이 80cm 를 의미한다.
    halfPix = linePix / 2 # halfPix은 실제 길이 40cm를 픽셀로 변환한 값
    pointArr = [] # path 점을 저장할 공간 
    
    for i in range(0, num):
        # 받은 이미지의 y축을 동일한 크기로 나눈다. 
        valY = image.shape[0] * (i + 1) / (num+1)
        valX = coefA * valY * valY + coefB * valY + coefC # valY 일때 x의 좌표
        slope = 2 * coefA * valY + coefB # 특정 점에서 접선의 기울기
        if slope != 0:
            revSlope = -1 / slope # 접선과 수직인 직선의 기울기
        else:
            revSlope = 1
        # 접선과 수직인 직선의 방정식
        # 2차함수가 x = ay^2 + by + c 형태이기 때문
        # x - x_0 = -1/m (y - y_0) 에 대입
        # x = revSlope * y - revSlope * valY + valX
        # 특정점을 원점으로 하는 길이가 halfPix 인 원의 방정식 
        # (x - x_0)^2 + (y - y_0)^2 = r^2 에 대입
        # (x - valX)^2 + (y - valY)^2 = halfPix^2
        # 위 두식을 연립하면 특정 점으로 부터 halfPix 만큼 떨어진 점을 얻을 수 있음
        meetY1 = valY + math.sqrt(halfPix*halfPix / (revSlope*revSlope + 1))
        meetY2 = valY - math.sqrt(halfPix*halfPix / (revSlope*revSlope + 1))
        # x좌표를 접선과 수직인 직선의 방정식에 대입해서 y값 계산
        meetX1 = revSlope * meetY1 - revSlope * valY + valX
        meetX2 = revSlope * meetY2 - revSlope * valY + valX
        # 한쪽 차선만 보인다는 가정
        # 우회전시 왼쪽차선, 좌회전시 오른쪽차선
        # 원과 직선의 교점은 2개인데, y값이 더 큰 값이 목표 path가 된다.
        if meetY1 > meetY2:
            pointArr.append([meetX1, meetY1])
        else:
            pointArr.append([meetX2, meetY2])
    # print (pointArr) # 디버깅 : 점 찍은 배열 확인
    return pointArr


def Detector(img):
    global fit_tmp, fit_true, fit_first

    img_tmp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
    lx = [] 
    ly = [] #

    rx = [] #
    ry = [] #

    mx = [] #
    my = [] #

    L_true = True #.
    R_true = True #

    c_left = 0 
    c_right = 0 #

    win_h = np.int(img.shape[0] / win_num) #


    not_zero = img.nonzero() #
    histogram= np.sum(img[100:,:], axis = 0)
    max_x = np.argmax(histogram)
    if(fit_first == False or fit_true == EMPTY):
        histogram= np.sum(img[100:,:], axis = 0)
        max_x = np.argmax(histogram)
        if max_x < 150:
            left_x_first = max_x
            his_x = max_x + 450
            right_x_first = np.argmax(histogram[his_x:]) + his_x
        elif max_x > 450:
            right_x_first = max_x
            his_x = max_x - 450
            left_x_first = np.argmax(histogram[:his_x])
        else:
            his_x = max_x - 150
            left_x_first = np.argmax(histogram[:his_x])
            his_x = max_x + 150
            right_x_first = np.argmax(histogram[his_x:]) + his_x
        fit_first = True
    else:
        coef_a = fit_tmp[0]
        coef_b = fit_tmp[1]
        coef_c = fit_tmp[2]
        #print("coef_a : {}, coef_b : {}, coef_c : {}".format(coef_a,coef_b,coef_c)) #for debugging ---> (coef_a)*x^2 + (coef_b)*x + (coef_c) 
        y_base = 450
        x_base = int(coef_a*y_base**2 + coef_b*y_base + coef_c)
        if fit_true == LEFT:
            left_x_first = x_base
            his_x = left_x_first + 300
            right_x_first = np.argmax(histogram[his_x:]) + his_x
        elif fit_true == RIGHT:
            right_x_first = x_base
            his_x = right_x_first - 300
            left_x_first = np.argmax(histogram[:his_x])
        else:
            his_x = x_base - 150
            left_x_first = np.argmax(histogram[:his_x])
            his_x = x_base + 150
            right_x_first = np.argmax(histogram[his_x:]) + his_x
    

    not_zero_y = np.array(not_zero[0]) #
    not_zero_x = np.array(not_zero[1]) 

    left_x_crr = left_x_first #
    right_x_crr = right_x_first #r

    for window in range(win_num):
        crr_win_y_h = img.shape[0] - window*win_h
        crr_win_y_l = img.shape[0] - (window+1)*win_h

        crr_win_left_x_low = left_x_crr - margin
        crr_win_left_x_hig = left_x_crr + margin

        crr_win_right_x_low = right_x_crr - margin
        crr_win_right_x_hig = right_x_crr + margin

        img_tmp = cv2.rectangle(img_tmp, (crr_win_right_x_low, crr_win_y_h), (crr_win_right_x_hig, crr_win_y_l), (0,0,255), 3)
        img_tmp = cv2.rectangle(img_tmp, (crr_win_left_x_low-1, crr_win_y_h), (crr_win_left_x_hig+1, crr_win_y_l), (0,255,0), 3)

        win_in_not_zero_l_x = ((not_zero_y >= crr_win_y_l)&(not_zero_y < crr_win_y_h)&(not_zero_x>=crr_win_left_x_low)&(not_zero_x < crr_win_left_x_hig)).nonzero()[0]
        win_in_not_zero_r_x = ((not_zero_y >= crr_win_y_l)&(not_zero_y < crr_win_y_h)&(not_zero_x>=crr_win_right_x_low)&(not_zero_x < crr_win_right_x_hig)).nonzero()[0]

        if(window == 0):
            if len(win_in_not_zero_l_x) > minpix_first:
                left_x_crr = np.int(np.mean(not_zero_x[win_in_not_zero_l_x]))
            if len(win_in_not_zero_r_x) > minpix_first: #
                right_x_crr = np.int(np.mean(not_zero_x[win_in_not_zero_r_x])) #
        else:
            
            if len(win_in_not_zero_l_x) > minpix: #)
                left_x_crr = np.int(np.mean(not_zero_x[win_in_not_zero_l_x])) #
                c_left += 1 #
                L_true = True #
            else: #
                L_true = False #
            #
            if len(win_in_not_zero_r_x) > minpix: 
                right_x_crr = np.int(np.mean(not_zero_x[win_in_not_zero_r_x])) #
                c_right += 1 #
                R_true = True #
            else: #
                R_true = False #
           
            if R_true and L_true: #
                lx.append(left_x_crr) #
                rx.append(right_x_crr) #
                mx.append((right_x_crr + left_x_crr)/2) #

                ly.append((crr_win_y_l + crr_win_y_h)/2) 
                ry.append((crr_win_y_l + crr_win_y_h)/2) #
                my.append((crr_win_y_l + crr_win_y_h)/2) #
            elif R_true == False and L_true == True:
                lx.append(left_x_crr) #
                ly.append((crr_win_y_l + crr_win_y_h)/2) #
            elif L_true == False and R_true == True:
                rx.append(right_x_crr) #
                ry.append((crr_win_y_l + crr_win_y_h)/2) 


   
    if((c_right >= min_win) and (c_left >= min_win)): 
        fit = np.polyfit(np.array(my),np.array(mx),2) #
        #
        fit_dir = MID #
    elif(c_right < min_win and c_left >= min_win): #
        fit = np.polyfit(np.array(ly),np.array(lx),2) 
        fit_dir = LEFT #

    elif(c_left < min_win and c_right >= min_win): #
        fit = np.polyfit(np.array(ry),np.array(rx),2) #f
        #
        fit_dir = RIGHT #
    else: #
        fit = [0, 0,320] #
        fit_dir = EMPTY #

    fit_true = fit_dir
    fit_tmp = fit

#######################################################
    if fit_dir != EMPTY and fit_dir != MID:
        arr = findPointCurve(img_tmp, fit_tmp[0], fit_tmp[1], fit_tmp[2], 550, 10)
        for i in arr:
            cv2.line(img_tmp, (int(i[0]), int(i[1])), (int(i[0]), int(i[1])), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    else:
        arr = findPointStraight(img_tmp, fit_tmp[0], fit_tmp[1], fit_tmp[2], 10)
        for i in arr:
            cv2.line(img_tmp, (int(i[0]), int(i[1])), (int(i[0]), int(i[1])), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    for i in arr:
        out.locate.append(path(x=i[0], y=i[1]))
    pub.publish(out)
#######################################################

    cv2.imshow('img_tmp', img_tmp)
    return fit_dir, fit #




while not rospy.is_shutdown():
    if cv_image.size != (640*480*3):
        continue
    cv2.imshow("original", cv_image)
    cv2.setMouseCallback('original',onMouse,cv_image)

    lane_img = lane_candi(cv_image)
    fit_dir, fit = Detector(lane_img)
    cv2.waitKey(1)
