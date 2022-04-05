import cv2
import numpy as np
from draw import *
from face_detection import *
from face_normalizer import *
from classifier import *



def face_recognition(img, classifier, visualize=True, unrecognized=True, probability=False, reverse=False):
    """A parir de l'image et du classifier (entraîné), détecte les visages.\n
    Renvoie la liste des visages reconnus.\n
    En option :\n
    - visualize : affichage des cadres de la détection de visage, écriture du nom de la personne (modifie la matrice img)\n
    - unrecognized : si pourcentage trop faible, pas de reconnaissance, le nom attribué est 'Unrecognized'\n
    - probability : affichage dans la console de la probailité de chaque personne
    - reverse : prend en compte le fait que l'image sera renversée""" 
    face_detection = main_face_detection(img)
    res = []
    for face_detection in main_face_detection(img):
        if visualize: draw_face_detection(img, face_detection, new=False)
        if len(face_detection["eyes_rect_list"]) != 0:
            hog_vector = main_face_normalizer(img, face_detection)
            hog_vector = np.array([hog_vector])
            name = classifier.predict(hog_vector)[0]
            if unrecognized:
                if max(classifier.predict_proba(hog_vector)[0]) < (1 / len(classifier.classes_)) * 2:
                    name = "Unrecognized"
            res.append(name)
            if probability:
                proba = np.round(classifier.predict_proba(hog_vector)[0], 3)
                print({classifier.classes_[i]: proba[i] for i in range(len(classifier.classes_))}, end=" "*50 + "\r")
            if visualize:
                draw_name(img, name, face_detection["face_rect"], rev=reverse)
    return res


def main():
    """Use the webcam for face recognition"""
    classifier = main_classifier()
    cap = cv2.VideoCapture(0)
    print("Starting of facial recognition...")
    print("\nType 'q' to quit\n")
    while True:
        _, frame = cap.read()
        face_recognition(frame, classifier, reverse=True)
        reverse(frame)
        cv2.imshow('Webcam', frame) # Affichage de l'image renversée
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()