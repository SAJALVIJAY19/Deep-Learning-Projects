from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

app = Flask(__name__)

# ======================== Load Saved Models ========================
lg = pickle.load(open('logistic_regression.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidfvectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# ======================== Preprocessing Function ========================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# ======================== Prediction Function ========================
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    return predicted_emotion, predicted_label

# ======================== Flask Route ========================
@app.route('/', methods=['GET', 'POST'])
def analyze_emotion():
    if request.method == 'POST':
        comment = request.form.get('comment')
        if not comment or not comment.strip():
            return render_template('index.html', sentiment=-1, comment='')

        predicted_emotion, label = predict_emotion(comment)
        return render_template('index.html', sentiment=label, emotion=predicted_emotion, comment=comment)

    return render_template('index.html', comment='')

# ======================== Run the App ========================
if __name__ == '__main__':
    app.run(debug=False)