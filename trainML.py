import pickle
#from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
#from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# Обучающие тексты и их ключевые слова
#train_texts = ["text1 with key1 key2", "text2 with key2 key3", "text3 with key1 key3"]
#train_labels = ["key1", "key2", "key1"]

file = 'markandnom.csv'
data = pd.read_csv(file, delimiter=';', index_col=False, encoding='utf8')
data = data.replace(np.nan, '')

data['nom'] = data['nom'].str.lower()
data['mark'] = data['mark'].str.lower()

train_texts = data['nom'].tolist()
train_labels = data['mark'].tolist()

cnt_claster = data['mark'].unique().shape[0]
#
# filetext = Path('nom.txt')
# lines = filetext.read_text(encoding='utf8').splitlines()
# train_texts = [x.lower() for x in lines]
#
# filetext = Path('mark.txt')
# lines = filetext.read_text(encoding='utf8').splitlines()
# train_labels = [x.lower() for x in lines]

# Создание объекта TfidfVectorizer для получения матрицы TF-IDF
stopwords = ["стиральная", "машина"]
vectorizer = TfidfVectorizer(stop_words=stopwords)
X_train = vectorizer.fit_transform(train_texts)

# Использование алгоритма KMeans для кластеризации текстов
kmeans = KMeans(n_clusters=cnt_claster, n_init='auto')
kmeans.fit(X_train)

# Использование алгоритма MultinomialNB для классификации текстов
clf = MultinomialNB()
clf.fit(X_train, train_labels)

pickle.dump(kmeans, open("save_kmean_nom_model.pkl", "wb"))
pickle.dump(clf, open("save_clf_nom_model.pkl", "wb"))
pickle.dump(vectorizer, open("save_vectorizer_nom_model.pkl", "wb"))