from face_detection import *
from db_path import *
from draw import *
import cv2
import numpy as np
from skimage.feature import hog

def affine_transform(img, face_rect, tp_point_list, destination_parametres = [0.4, 0.3]):
    """destination_parametres : [distance yeux - haut du cadre (en % de height), distance bord gauche + oeil gauche (en % de width)]"""
    source = np.float32(tp_point_list)
    p1, p2 = destination_parametres
    x, y, width, height = face_rect
    left_eye = [x + p2*width, y + p1*height]
    right_eye = [x + (1-p2)*width, y + p1*height]
    third_point = [x + 0.5*width, y + height]
    destination = [left_eye, right_eye, third_point]
    destination = [[int(x1), int(x2)] for x1, x2 in destination]
    destination = np.float32(destination)
    affine = cv2.getAffineTransform(source, destination)
    return cv2.warpAffine(img, affine, img.shape[1::-1], flags=cv2.INTER_LINEAR)

def resize(img, face_rect, size=256):
    x, y, width, height = face_rect
    sub_image = img[y:y+height, x:x+width] 
    return cv2.resize(sub_image, (size, size))

def hog_representation(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1,1)):
    """Computes the HOG representation of the image"""
    image_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hog_vector, hog_image = hog(image_gray, orientations, pixels_per_cell, cells_per_block, visualize=True)
    #show(hog_image)
    return hog_vector


def main_face_normalizer(img, face_detection, visualize=False):
    """Renvoie le vecteur hog à partir de l'image et du dictionnaire face detection, en option affiche l'image intermédiaire 256 x 256"""
    sub_image = resize(affine_transform(img, face_detection["face_rect"], face_detection["three_points_list"]), face_detection["face_rect"])
    if visualize: show(sub_image)
    return hog_representation(sub_image)

if __name__ == "__main__":
    for name, path in nav_random(5):
        img = cv2.imread(path)
        for face_detection in main_face_detection(img):
            main_face_normalizer(img, face_detection, visualize=True)