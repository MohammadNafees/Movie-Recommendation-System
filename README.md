# 🎬 Movie Recommendation System using NLP

This project is a **Movie Recommendation System** that leverages **Natural Language Processing (NLP)** techniques to suggest similar movies based on a given input. The system uses movie plot overviews and other metadata to recommend movies with similar content.

## 🚀 Features

- Content-based movie recommendations using plot overviews
- Text preprocessing with NLP (tokenization, stopwords removal, TF-IDF)
- Cosine similarity to find similar movies
- Built with Python and popular NLP libraries

## 📂 Project Structure

movie-recommendation-system-nlp/
├── data/
│ └── movies.csv
├── notebooks/
│ └── exploration.ipynb
├── requirements.txt
└── README.md


## 📊 Dataset

We use the **IMDB 5000 Movie Dataset**, which includes:
- Movie titles
- Plot overviews
- Genres
- Keywords
- Cast and crew

> Dataset source: Kaggle

## 🛠️ Tech Stack

- Python 3.x
- Pandas & NumPy
- Scikit-learn
- NLTK / spaCy

## 🧠 How It Works

1. **Preprocessing**:
   - Clean and normalize movie overviews
   - Tokenization, stopword removal, stemming/lemmatization
2. **Vectorization**:
   - Convert text data into numerical vectors using TF-IDF
3. **Similarity Calculation**:
   - Use cosine similarity to find top N similar movies
4. **Recommendation**:
   - Input a movie title and get similar recommendations

              precision    recall  f1-score   support

    Positive       0.75      0.76      0.75      4993
    Negative       0.75      0.75      0.75      5007

    accuracy                           0.75     10000
   macro avg       0.75      0.75      0.75     10000
weighted avg       0.75      0.75      0.75     10000

              precision    recall  f1-score   support

    Positive       0.75      0.76      0.75      4993
    Negative       0.75      0.74      0.75      5007

    accuracy                           0.75     10000
   macro avg       0.75      0.75      0.75     10000
weighted avg       0.75      0.75      0.75     10000

<img width="611" height="314" alt="image" src="https://github.com/user-attachments/assets/1350bec6-dd71-4fa7-9ff2-29af04980fc0" />


<img width="611" height="314" alt="image" src="https://github.com/user-attachments/assets/5f857d3e-00a1-4859-a04f-837a93799c7c" />

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mohammadnafees/movie-recommendation-system.git
cd movie-recommender-nlp
pip install -r requirements.txt
```

📌 Future Improvements
Adding hybrid filtering with collaborative methods

Using deep learning models like BERT for embeddings

Adding user login and personalized recommendations

Deploying as a web app with a clean UI

🤝 Contributing
Contributions are welcome! Feel free to fork this repo and submit a pull request.

📜 License
This project is open-source and available under the MIT License.
