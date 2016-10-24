from sklearn.externals import joblib
question = raw_input(":> Ingresa la pregunta: ")
clf = joblib.load('clf.pkl')
vectorizer = joblib.load('vectorizer.pkl')
selector = joblib.load('selector.pkl')
question = vectorizer.transform([question])
question = selector.transform(question).toarray()

print clf.predict(question)
