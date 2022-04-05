import cv2
from os import listdir, makedirs, rename, rmdir, remove
from os.path import exists
from db_path import *
from face_detection import *
from face_normalizer import *

def create_database(add=False):
    """Compute the preprocessed images to vector representation,return the set array (option add: use the data in folder Add)"""
    data_x = []  # List of normalised faces
    data_y = []  # List of labels
    if add: print("Adding data from folder 'Add'")
    else: print("Creation of a new data sample")
    print("Computing the hog vector of each image...")
    if add: folder = "Add"
    else: folder = "Dataset"
    d, s = count(folder=folder)
    print(str(s) + " images found, from " + str(len(d)) + " different people")
    done = {x: 0 for x in d.keys()}
    for name, path in nav_all(folder=folder):
        done[name] += 1
        fraction = " [" + str(done[name]) + "/" + str(d[name]) + "]"
        n = int((done[name]-1) / d[name] * 20)
        bar = " [" + "X"*n + "_"*(20-n) + "] "
        file = path.split(sep="/")[-1]
        if len(file) > 25: file = file[:20] + "..." 
        print("Computing the list of normalised faces for " + name + "..." + fraction + bar + file, end=" "*100 + "\r")
        img = cv2.imread(path)
        face_detection = main_face_detection(img)
        if len(face_detection) != 1:
            print("[Fail] - [" + name + "] Number of detected face : " + str(len(face_detection)) + " | File name : " + file, end=" "*100 + "\n")
        else:
            face_detection = face_detection[0]
            if len(face_detection["eyes_rect_list"]) == 0:
                print("[Fail] - [" + name + "] Fail in eyes recognition | File name : " + file, end=" "*100 + "\n")
            else:
                hog_vector = main_face_normalizer(img, face_detection)
                data_x.append(hog_vector)
                data_y.append(name)
                if add:
                    if not exists("./Dataset/" + name): makedirs("./Dataset/" + name)
                    rename(path, path.replace("Add", "Dataset"))
                    if len(listdir("./Add/" + name)) == 0: rmdir("./Add/" + name)
        if done[name] == d[name]: print("Computing the list of normalised faces for " + name + "..." + fraction + " [" + "X"*20 + "]", end=" "*100 + "\n")

    print("Computations finished ! [success : " + str(len(data_x)) + "/" + str(s) + "]")
    return np.array(data_x), np.array(data_y)

def save_database(data_x, data_y):
    """
    Saves the database
    :param data_x: the inputs matrix
    :param data_y: the output matrix
    Erases the saved classifier (if exists) to force new calculations
    """
    np.save("./Save/dx.npy", data_x)
    np.save("./Save/dy.npy", data_y)
    if "fitted_classifier.sav" in listdir("./Save"): remove("./Save/fitted_classifier.sav")

def load_database():
    """
    Loads the database
    :return: the input and output matrix
    """
    data_x = np.load("./Save/dx.npy")
    data_y = np.load("./Save/dy.npy")
    return data_x, data_y

def init_database():
    """Initialise la base de données à partir des fichiers présents dans dataset (créer ou écrase la précédente)"""
    data_x, data_y = create_database()
    save_database(data_x, data_y)

def delete_database():
    """Efface la base de données"""
    for file in listdir("./Save"):
        remove("./Save/" + file)

def add_data():
    """Add images in folder 'Add' (in the same format as in 'Dataset') to the data sample (and save it), then move the image from 'Add' folder to 'Dataset' folder (if no error)\n
    It will compute the data sample from 'Dataset' folder if not done\n
    It will also erase the saved classifier (if exists) to force new calculations"""
    if count(folder="Add")[1] > 0:
        files = listdir("./Save")
        if not ("dx.npy" in files and "dy.npy" in files):
            init_database()
        old_x, old_y = load_database()
        new_x, new_y = create_database(add=True)
        save_database(np.concatenate((old_x, new_x)), np.concatenate((old_y, new_y)))


def main_db_management(add=False):
    """Initialise la base de données si il n'y en a pas, sinon ne fait rien"""
    files = listdir("./Save")
    if not ("dx.npy" in files and "dy.npy" in files):
        init_database()
    if add: add_data()

if __name__ == "__main__":
    main_db_management(add=True)