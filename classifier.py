from db_management import *
from sklearn.neural_network import MLPClassifier
import pickle

# Cacher les warnings
import warnings
warnings.filterwarnings('ignore')

def save_classifier(classifier):
    """Saves the fitted classifier"""
    pickle.dump(classifier, open("./Save/fitted_classifier.sav", 'wb'))

def load_classifier():
    """Loads the fitted classifier"""
    return pickle.load(open("./Save/fitted_classifier.sav", 'rb'))

def main_classifier():
    """Créé et entraîne le classifier (si cela a déjà été fait, le charge depuis le dossier save)"""
    files = listdir("./Save")
    if not ("fitted_classifier.sav" in files):
        main_db_management()
        X, Y = load_database()
        print("Classifier creation...")
        classifier = MLPClassifier(alpha=1, max_iter=1000, solver='sgd', activation='relu', hidden_layer_sizes=(100,))
        print("Classifier fitting...")
        classifier.fit(X, Y)
        save_classifier(classifier)
    else:
        classifier = load_classifier()
    return classifier


if __name__ == '__main__':
    main_classifier()