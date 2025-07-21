# EmotiWave: A Text Emotion Recognition System 🚀

EmotiWave is an advanced emotion detection system that leverages **LSTM deep learning** and **machine learning models** to classify human emotions from textual data. Designed with a sleek web interface powered by **Flask**, it allows users to input free text and receive real-time emotion predictions like Joy, Anger, Love, and more.

https://drive.google.com/file/d/1DZhOn_DyWkFonfRKokzqaNDbebOhOwC7/view?usp=sharing

---

## 🧠 Models Used

- 📦 **TF-IDF + Logistic Regression** — Traditional ML pipeline for text emotion classification.
- 🧠 **LSTM (Long Short-Term Memory)** — Deep learning model trained for sequential emotion recognition (included as `model.h5`).

---

## 🔧 Tech Stack

| Layer        | Technology              |
|-------------|--------------------------|
| Frontend    | HTML5, CSS3,             |
| Backend     | Python, Flask            |
| ML Models   | Scikit-learn, Keras      |
| NLP Tools   | NLTK (stopwords, stemming) |
| Deployment  | Localhost / Cloud-Ready  |

---

## ⚙️ How It Works

1. User inputs a sentence or paragraph.
2. Text is preprocessed using NLTK (tokenization, stopword removal, stemming).
3. The cleaned text is passed to the **TF-IDF + Logistic Regression** pipeline.
4. The system predicts one of the 6 emotions:
   - 😠 Anger
   - 😨 Fear
   - 😊 Joy
   - 💖 Love
   - 😢 Sadness
   - 😲 Surprise

---

## 📁 Project Structure

EmotiWave/
│
├── app.py # Flask web app
├── EmotiWave.ipynb # Notebook for model training/testing
├── template/
│ └── index.html # Frontend HTML page
├── label_encoder.pkl # LabelEncoder used for decoding model output
├── tfidfvectorizer.pkl # TF-IDF vectorizer for ML model
├── logistic_regression.pkl # Trained logistic regression model
├── model.h5 # Trained LSTM model (optional)
├── data/ # Raw / cleaned data (optional)
└── vocab_info.pkl # Vocab/token info for LSTM model



---

## 🚀 Running the App Locally

### 🔹 Step 1: Clone the repo
```bash
git clone https://github.com/your-username/EmotiWave.git
cd EmotiWave

### 🔹 Step 2: Install dependencies
pip install -r requirements.txt

### 🔹 Step 3: Run the app
python app.py


