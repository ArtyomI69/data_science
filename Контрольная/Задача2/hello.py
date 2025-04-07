import streamlit as st
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from gensim.models import Word2Vec
import pymorphy3
import nltk

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(layout="wide")
st.title("Анализ текстов песен")

# Вкладки
tab_wordcloud, tab_tfidf, tab_tsne = st.tabs(["WordCloud", "TF-IDF", "t-SNE: 15 частых слов"])


# Загрузка текстов из папки
@st.cache_data
def load_texts_from_folder(folder):
    texts = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), encoding="utf-8") as f:
                texts.append(f.read())
    return texts


# Предобработка
morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))


def preprocess(text):
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    text = re.sub(r"[^а-яё\s]", " ", text)
    tokens = text.split()
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    return [t for t in lemmas if t not in russian_stopwords and len(t) > 1]


# Загружаем тексты
texts = load_texts_from_folder("C:\\Users\\User\\OneDrive\\Desktop\\Анализ больших данных\\Контрольная\\Задача2\\песни")
processed = [preprocess(t) for t in texts]
joined = [" ".join(tokens) for tokens in processed]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(joined)
features = vectorizer.get_feature_names_out()
tfidf_sums = np.array(tfidf.sum(axis=0)).flatten()
top_indices = tfidf_sums.argsort()[-10:][::-1]
top_words = [features[i] for i in top_indices]

# Word2Vec
model = Word2Vec(sentences=processed, vector_size=100, window=5, min_count=1, workers=4)

with tab_wordcloud:
    st.header("Облако слов")
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        colormap="autumn"
    ).generate(" ".join(joined))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

with tab_tfidf:
    st.header("Топ-10 слов по TF-IDF")
    st.write("Наиболее значимые слова по TF-IDF:")
    for i, word in enumerate(top_words, 1):
        st.markdown(f"**{i}. {word}**")

with tab_tsne:
    st.header("t-SNE: визуализация 15 частых слов")
    common = [w for w, _ in Counter(" ".join(joined).split()).most_common(15)]
    vectors = [model.wv[w] for w in common if w in model.wv]

    if len(vectors) >= 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        coords = tsne.fit_transform(np.array(vectors))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, word in enumerate(common):
            if word in model.wv:
                ax.scatter(coords[i, 0], coords[i, 1])
                ax.annotate(word, (coords[i, 0] + 0.5, coords[i, 1] + 0.5))
        ax.set_title("t-SNE: 15 частых слов")
        st.pyplot(fig)
    else:
        st.warning("Недостаточно слов для t-SNE визуализации.")
