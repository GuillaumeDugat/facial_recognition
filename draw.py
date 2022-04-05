import cv2
import numpy as np

def show(img, name =""):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

def draw_rectangles(img, rect_lists, color=(0,0,255)):
    """A partir d'une liste de rectangle, dessine les rectangles sur l'image"""
    for x, y, width, height in rect_lists:
        cv2.rectangle(img, (x,y), (x+width, y+height), color)

def draw_points(img, point_lists, color=(0,0,255), size=4):
    """A partir d'une liste de points, dessine les rectangles sur l'image"""
    for x, y in point_lists:
        cv2.rectangle(img, (x-size, y-size), (x+size, y+size), color)

def draw_face_detection(img, face_detection, new=True):
    """A partir du dictionnaire renvoyé par face_detection, trace les cadres du visage et des yeux ainsi que que les trois points utilisés pour la transformation affine.
    Ne modifie pas l'image actuelle, en créer une nouvelle"""
    if new: res = np.copy(img)
    else: res = img
    draw_rectangles(res, [face_detection["face_rect"]])
    draw_rectangles(res, face_detection["eyes_rect_list"], color=(255,0,0))
    draw_points(res, face_detection["three_points_list"], color=(0,255,0))
    return res

def reverse(img):
    """Symétrie horizontal de la matrice associée (modification en place)"""
    img[:] = img[:,::-1]

def draw_name(img, name, face_rect, color=(0,0,255), rev=False):
    """Write name next to the face rectangle"""
    x, y, width, _ = face_rect
    if rev:
        x = img.shape[1] - (x + width)
        reverse(img)
    cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    if rev: reverse(img)