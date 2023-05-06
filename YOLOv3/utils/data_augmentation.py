import torch
import torch.nn.functional as F
import numpy as np
import cv2



def horizontal_flip(img, filled_labels):
    h,w,_ = img.shape
    img = cv2.flip(img,1)
    x_cn, y_cn, w_box_n, h_box_n = filled_labels[0][1:]
    # x_min,y_min,x_max,y_max = filled_labels
    #
    # new_xmin = w-x_min
    # new_ymin = y_min
    # new_xmax = w-x_max
    # new_ymax = y_max

    new_x_cn = 1-x_cn
    new_y_cn = y_cn
    new_w_box_n = w_box_n
    new_h_box_n = h_box_n
    filled_labels[0][1] = new_x_cn
    filled_labels[0][2] = new_y_cn
    filled_labels[0][3] = new_w_box_n
    filled_labels[0][4] = new_h_box_n



    # filled_labels = [new_x_cn, new_y_cn,new_w_box_n,new_h_box_n]

    return img, filled_labels

#
#
# def horizontal_flip(img, filled_labels):
#     h,w,_ = img.shape
#     img = cv2.flip(img,1)
#     x_cn, y_cn, w_box_n, h_box_n = filled_labels
#     # x_min,y_min,x_max,y_max = filled_labels
#     #
#     # new_xmin = w-x_min
#     # new_ymin = y_min
#     # new_xmax = w-x_max
#     # new_ymax = y_max
#     new_x_cn = 1-x_cn
#     new_y_cn = y_cn
#     new_w_box_n = w_box_n
#     new_h_box_n = h_box_n
#
#
#     filled_labels = [new_x_cn, new_y_cn,new_w_box_n,new_h_box_n]
#     return img, filled_labels
#
#
def vertical_flip(img, filled_labels):
    h,w,_ = img.shape
    img = cv2.flip(img,0)
    x_cn, y_cn, w_box_n, h_box_n = filled_labels[0][1:]
    # x_min,y_min,x_max,y_max = filled_labels
    #
    # new_xmin = w-x_min
    # new_ymin = y_min
    # new_xmax = w-x_max
    # new_ymax = y_max
    new_x_cn = x_cn
    new_y_cn = 1-y_cn
    new_w_box_n = w_box_n
    new_h_box_n = h_box_n
    filled_labels[0][1] = new_x_cn
    filled_labels[0][2] = new_y_cn
    filled_labels[0][3] = new_w_box_n
    filled_labels[0][4] = new_h_box_n


    # filled_labels = [new_x_cn, new_y_cn,new_w_box_n,new_h_box_n]
    return img, filled_labels
#
def rotate90Deg(filled_labels): # just passing width of image is enough for 90 degree rotation.
   # x_min,y_min,x_max,y_max = filled_labels
   x_cn, y_cn, w_box_n, h_box_n = filled_labels[0][1:]
   # new_xmin = y_min
   # new_ymin = w-x_max
   # new_xmax = y_max
   # new_ymax = w-x_min
   new_x_cn = y_cn
   new_y_cn = 1-x_cn
   new_w_box_n = h_box_n
   new_h_box_n = w_box_n
   filled_labels[0][1] = new_x_cn
   filled_labels[0][2] = new_y_cn
   filled_labels[0][3] = new_w_box_n
   filled_labels[0][4] = new_h_box_n


   return filled_labels

def rotate_90(img,filled_labels):
    h,w,_ = img.shape
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    filled_labels = rotate90Deg(filled_labels)
    return img, filled_labels


def rotate_180(img,filled_labels):
    h,w,_ = img.shape
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    filled_labels = rotate90Deg(filled_labels)
    h,w,_ = img.shape
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    filled_labels = rotate90Deg(filled_labels)
    # img =  rotate_90(img,filled_labels)[0]
    # filled_labels = rotate_90(img,filled_labels)[1]
    # h,w,_ = img.shape
    # img =  rotate_90(img,filled_labels)[0]
    # filled_labels = rotate_90(img,filled_labels)[1]

    return img, filled_labels

def rotate_270(img,filled_labels):
    h,w,_ = img.shape
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    filled_labels = rotate90Deg(filled_labels)
    h,w,_ = img.shape
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    filled_labels = rotate90Deg(filled_labels)
    h,w,_ = img.shape
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    filled_labels = rotate90Deg(filled_labels)
    # img =  rotate_90(img,filled_labels)[0]
    # filled_labels = rotate_90(img,filled_labels)[1]
    # h,w,_ = img.shape
    # img =  rotate_90(img,filled_labels)[0]
    # filled_labels = rotate_90(img,filled_labels)[1]

    return img, filled_labels


def augmentation(input_img, filled_labels):
    # horizontal flip
    # img = input_img
    # labels = filled_labels
    ran_int_h = np.random.randint(0, 2)
    # print(ran_int_h)
    if ran_int_h == 1:
       input_img, filled_labels = horizontal_flip(input_img, filled_labels)

    ran_int_v = np.random.randint(0, 2)
    # print(ran_int_v)
    if ran_int_v == 1:
       input_img, filled_labels = vertical_flip(input_img, filled_labels)

    angle = np.random.choice([0,90,180,270],1)[0]
    # print(angle)
    if angle == 270:
       input_img, filled_labels = rotate_270(input_img, filled_labels)
    elif angle ==180:
        input_img, filled_labels = rotate_180(input_img, filled_labels)
    elif angle ==90:
        input_img, filled_labels = rotate_90(input_img, filled_labels)
    elif angle ==0:
        pass

    return input_img, filled_labels


#
#
# def rotate_90(img,filled_labels):
#     h,w,_ = img.shape
#     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     filled_labels = rotate90Deg(filled_labels)
#     return img, filled_labels
#
# def rotate_180(img,filled_labels):
#     h,w,_ = img.shape
#     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     filled_labels = rotate90Deg(filled_labels)
#     h,w,_ = img.shape
#     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     filled_labels = rotate90Deg(filled_labels)
#     # img =  rotate_90(img,filled_labels)[0]
#     # filled_labels = rotate_90(img,filled_labels)[1]
#     # h,w,_ = img.shape
#     # img =  rotate_90(img,filled_labels)[0]
#     # filled_labels = rotate_90(img,filled_labels)[1]
#
#     return img, filled_labels
#
# def rotate_270(img,filled_labels):
#     h,w,_ = img.shape
#     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     filled_labels = rotate90Deg(filled_labels)
#     h,w,_ = img.shape
#     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     filled_labels = rotate90Deg(filled_labels)
#     h,w,_ = img.shape
#     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     filled_labels = rotate90Deg(filled_labels)
#     # img =  rotate_90(img,filled_labels)[0]
#     # filled_labels = rotate_90(img,filled_labels)[1]
#     # h,w,_ = img.shape
#     # img =  rotate_90(img,filled_labels)[0]
#     # filled_labels = rotate_90(img,filled_labels)[1]
#
#     return img, filled_labels
#
# def augmentation(input_img, filled_labels):
#     # horizontal flip
#     # img = input_img
#     # labels = filled_labels
#     ran_int_h = np.random.randint(0, 2)
#     print(ran_int_h)
#     if ran_int_h == 1:
#        input_img, filled_labels = horizontal_flip(input_img, filled_labels)
#
#     ran_int_v = np.random.randint(0, 2)
#     print(ran_int_v)
#     if ran_int_v == 1:
#        input_img, filled_labels = vertical_flip(input_img, filled_labels)
#
#     angle = np.random.choice([0,90,180,270],1)[0]
#     print(angle)
#     if angle == 270:
#        input_img, filled_labels = rotate_270(input_img, filled_labels)
#     elif angle ==180:
#         input_img, filled_labels = rotate_180(input_img, filled_labels)
#     elif angle ==90:
#         input_img, filled_labels = rotate_90(input_img, filled_labels)
#     elif angle ==0:
#         pass
#
#     return input_img, filled_labels