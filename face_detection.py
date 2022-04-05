import cv2
from os import listdir
from db_path import *
from draw import *

eye_cascade_classifier = cv2.CascadeClassifier("./Cascade_Classifiers/haarcascade_eye.xml")
face_cascade_classifier = cv2.CascadeClassifier("./Cascade_Classifiers/haarcascade_frontalface_alt.xml")

def find_faces(img):
    """Prend une image en entrée (sous forme de matrice) et renvoie la liste des cadres des détection de visage"""
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, _, _ = face_cascade_classifier.detectMultiScale3(image_gray, 1.1, 5, flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True)
    return faces
    
def find_eyes(img, rect):
    """Recherche les yeux dans la portion de l'image rect, renvoie la réponse sous forme d'une liste de des rectangles des yeux ainsi que la liste de confiance"""
    x, y, width, height = rect
    sub_image = img[y:y+height, x:x+width]
    image_gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
    eyes, _, confiance = eye_cascade_classifier.detectMultiScale3(image_gray, 1.1, 5, flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True)
    for eye in eyes:
        eye[0] += x
        eye[1] += y
    return eyes, confiance

def filter_eyes(eyes, confiance):
    """Ne garde que deux yeux, ceux de confiance maximale, élément 0 : gauche, élément 1 : droit, le résultat est la liste (de taille 2) des rectangles.
    Si la liste est vide : c'est qu'il y a 0 ou 1 oeil."""
    if len(eyes) < 2:
        return []
    else:
        confiance = list(confiance)
        c0, c1 = sorted(confiance)[-2:]
        i0, i1 = confiance.index(c0), confiance.index(c1)
        eye0, eye1 = eyes[i0], eyes[i1]
        if eye0[0] < eye1[0]:
            return [eye0, eye1]
        else:
            return [eye1, eye0]

def eye_position(rect):
    return [rect[0] + rect[2] // 2, rect[1] + rect[3] // 2]

def three_points(face_rect, eyes_rect_list):
    left_eye = eye_position(eyes_rect_list[0])
    right_eye = eye_position(eyes_rect_list[1])
    lx, ly = left_eye
    rx, ry = right_eye
    # le troisième point (coordonnées tx, ty) est placé à l'intersection de la médiatrice des yeux et du cadre de la face (en bas)
    assert lx != rx or ly != ry
    mx, my = (lx + rx) / 2, (ly + ry) / 2 # calcul des coordonnées du milieu
    if rx != lx:
        ty = face_rect[1] + face_rect[3] # bordure du bas
        tx = -(ty - my) * (ry - ly) / (rx - lx) + mx
        if tx >= face_rect[0] and tx < face_rect[0] + face_rect[2]:
            return [left_eye, right_eye, [int(tx), int(ty)]]
    tx1 = face_rect[0] # bordure de gauche
    ty1 = -(tx1 - mx) * (rx - lx) / (ry - ly) + my
    tx2 = face_rect[0] + face_rect[3] # bordure de droite
    ty2 = -(tx2 - mx) * (rx - lx) / (ry - ly) + my
    if ty1 < ty2:
        return [left_eye, right_eye, [int(tx2), int(ty2)]]
    else:
        return [left_eye, right_eye, [int(tx1), int(ty1)]]


def main_face_detection(img, visualize=False):
    """Renvoie pour chaque visage détecté (sous forme de liste), le rectangle associé à la face, la liste des rectangles des deux yeux (vide si échec),
    la liste des trois points (oeil gauche, oeil droit, menton) (vide si échec)"""
    res = []
    face_rect_list = find_faces(img)
    for face_rect in face_rect_list:
        new_face = {"face_rect" : face_rect}
        eyes_rect_list, confiance = find_eyes(img, face_rect)
        eyes_rect_list = filter_eyes(eyes_rect_list, confiance)
        new_face["eyes_rect_list"] = eyes_rect_list
        if len(eyes_rect_list) == 2:
            tp_point_list = three_points(face_rect, eyes_rect_list)
            new_face["three_points_list"] = tp_point_list
        else:
            new_face["three_points_list"] = []
        res.append(new_face)
        if visualize: show(draw_face_detection(img, new_face))
    return res

if __name__ == "__main__":
    for name, path in nav_test(): #nav_random(5):
        img = cv2.imread(path)
        main_face_detection(img, visualize=True)