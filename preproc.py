import os
import numpy
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

features = []
labels = []

def abre_txt(path_trainingSet):
    global features
    global labels
#    for i in os.listdir(path_trainingSet):
    file = open("trainingSet.txt", "rb")
    reader = file.readlines()
    features_aux = [line.replace("\n","").split(":")[1] for line in reader]
    features = features + features_aux
    labels_aux = [line.replace("\n","").split(":")[0] for line in reader]
    labels  = labels + labels_aux
    print labels

def preprocesa(features, labels):
    features_train, features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,test_size=0.1,random_state=42)
    vectorizer = TfidfVectorizer(sublinear_tf = True,  max_df = 0.5, stop_words = "english")
    features_train = vectorizer.fit_transform(features_train)
    features_test = vectorizer.transform(features_test)
    joblib.dump(vectorizer, 'vectorizer.pkl')

    selector = SelectPercentile(f_classif, percentile = 10)
    selector.fit(features_train, labels_train)
    joblib.dump(selector, 'selector.pkl')
    features_test =selector.transform(features_test).toarray()
    features_train = selector.transform(features_train).toarray()

    return features_train, features_test, labels_test, labels_train

def main():
    abre_txt("trainingSet.txt")
    features_train,features_test,labels_test,labels_train = preprocesa(features,labels)
    clf = SVC(kernel = "linear")
    clf.fit(features_train, labels_train)
    joblib.dump(clf, 'clf.pkl')
    pred = clf.predict(features_test)
    print accuracy_score(pred, labels_test)

if __name__ == '__main__':
    main()
