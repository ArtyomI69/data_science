
import re
import os
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

# Загрузка текстов из файлов
def load_texts_from_folder(folder):
    texts = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), encoding="utf-8") as f:
                texts.append(f.read())
    return texts

songs = load_texts_from_folder("песни")
texts = songs
labels = ["song"] * len(songs)

# Предобработка текста
morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))
def preprocess(text):
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    text = re.sub(r"[^а-яё\s]", " ", text)
    tokens = text.split()
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    return [t for t in lemmas if t not in russian_stopwords and len(t) > 1]

processed = [preprocess(t) for t in texts]
joined = [" ".join(tokens) for tokens in processed]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(joined)
features = vectorizer.get_feature_names_out()
tfidf_sums = np.array(tfidf.sum(axis=0)).flatten()
top_indices = tfidf_sums.argsort()[-10:][::-1]
top_words = [features[i] for i in top_indices]
print("\nТоп-10 слов по TF-IDF:", top_words)

# WordCloud
wordcloud = WordCloud(
    width=800,
    height=800,
    background_color="white",
    colormap="autumn",
    contour_width=0  # убираем контур
).generate(" ".join(joined))
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud", fontsize=16)
plt.show()

# Word2Vec
model = Word2Vec(sentences=processed, vector_size=100, window=5, min_count=1, workers=4)
if "война" in model.wv:
    print("\nПохожие слова к 'война':")
    for w, s in model.wv.most_similar("война", topn=5):
        print(f" - {w} ({s:.3f})")

# t-SNE
common = [w for w, _ in Counter(" ".join(joined).split()).most_common(15)]
vectors = [model.wv[w] for w in common if w in model.wv]
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
coords = tsne.fit_transform(np.array(vectors))

plt.figure(figsize=(8, 6))
for i, word in enumerate(common):
    if word in model.wv:
        plt.scatter(coords[i, 0], coords[i, 1])
        plt.annotate(word, (coords[i, 0] + 0.5, coords[i, 1] + 0.5))
plt.title("t-SNE: 15 частых слов")
plt.show()
