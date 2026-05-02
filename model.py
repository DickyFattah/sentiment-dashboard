from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_model():
    # dummy training (ganti dengan model kamu nanti)
    texts = ["good product", "bad quality"]
    labels = ["positive", "negative"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    return (model, vectorizer)

def predict_sentiment(model_bundle, text):
    model, vectorizer = model_bundle
    X = vectorizer.transform([text])
    return model.predict(X)[0]