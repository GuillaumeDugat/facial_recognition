from os import listdir
import random as rnd

def labels(folder="Dataset"):
    return listdir("./" + folder)

def nav_all(folder="Dataset"):
    """Générateur pour parcourir toutes les photos"""
    for name in labels(folder=folder):
        photo_list = listdir("./" + folder + "/" + name)
        for photo in photo_list:
            path = "./" + folder + "/" + name + "/" + photo
            yield name, path

def nav_random(n):
    """Générateur pour parcourir toutes les photos"""
    for _ in range(n):
        name = rnd.choice(labels())
        photo_list = listdir("./Dataset/" + name)
        photo = rnd.choice(photo_list)
        path = "./Dataset/" + name + "/" + photo
        yield name, path

def nav_test():
    for photo in listdir("./Test"):
        path = "./Test/" + photo
        yield "no name", path

def count(folder="Dataset"):
    d = {name: len(listdir("./" + folder + "/" + name)) for name in labels(folder=folder)}
    s = sum([d[name] for name in d])
    return d, s