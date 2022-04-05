from db_management import *
import random as rnd
from time import time

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Cacher les warnings
import warnings
warnings.filterwarnings('ignore')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

def classifiers():
    classifiers_list = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000, solver='sgd', activation='relu', hidden_layer_sizes=(100,)),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    return classifiers_list

def main_classifier_comparison(n=5, p=0.2):
    """Entrainement puis test des classifiers, affichage des scores de chacun
    n : nombre de répétition, on fait ensuite la moyenne
    p : proportion de la base de données utilisée pour le test"""
    print("Starting classifiers comparison...")
    X, Y = load_database()
    index_dict = {} # Dictionnaire contenant pour chaque label la liste des indices des éléments dans Y
    for i in range(len(Y)):
        label = Y[i]
        if not (label in index_dict): index_dict[label] = []
        index_dict[label].append(i)
    score = []
    time_score = []
    for k in range(n):
        print("Step " + str(k+1) + "/" + str(n), end=" "*50 + "\r")
        training_index_list, test_index_list = [], []
        for label in index_dict:
            index_list = index_dict[label].copy()
            rnd.shuffle(index_list)
            l = len(index_list)
            size = int(p*l) # taille de la population test
            assert size > 0 and l - size > 0
            test_index_list += index_list[:size]
            training_index_list += index_list[size:]
        X_training, Y_training = X[training_index_list], Y[training_index_list]
        X_test, Y_test = X[test_index_list], Y[test_index_list]
        step_score = []
        step_time = []
        j = 0
        for classifier in classifiers():
            print("Step " + str(k+1) + "/" + str(n) + " | " + names[j] + "...", end=" "*50 + "\r")
            j += 1
            t0 = time()
            classifier.fit(X_training, Y_training)
            s = classifier.score(X_test, Y_test)
            step_time.append(time() - t0)
            step_score.append(s)
        score.append(step_score)
        time_score.append(step_time)
        print("Step " + str(k+1) + "/" + str(n) + (" | time : %.2f s" % sum(step_time)))
    mean_score = np.mean(score, axis=0)
    mean_score = np.round(mean_score, 3)
    res = {names[i] : mean_score[i] for i in range(len(names))}
    mean_time = np.mean(time_score, axis=0)
    mean_time = np.round(mean_time, 3)
    res_time = {names[i] : mean_time[i] for i in range(len(names))}
    print("Result :\nScore : " + str(res) + "\nTime (in s) : " + str(res_time))
    return res, res_time

if __name__ == "__main__":
    main_classifier_comparison()