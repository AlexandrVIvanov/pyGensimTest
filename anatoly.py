import pandas as pd
import numpy as np

from tqdm.auto import tqdm, trange
import nltk
import re
import pprint

file = 'markandnom.csv'

data = pd.read_csv(file, delimiter=';', index_col=False)

mark = data['mark'].unique()

brands = mark.tolist()

cnt_claster = mark.shape[0]

print("количество кластеров: " + str(cnt_claster))

df_res = pd.DataFrame()

for brand in tqdm(brands):
    #     print(brand)

    df_topic = data[data['mark'] == brand]

    #     print(data[data['mark'] == brand])

    df_res = df_res.append(df_topic, ignore_index=True)

texts = df_res['nom']

nltk.download('punkt')

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("russian", "english")


def token_and_stem(text):
    tokens = ["Принтер", "Холодильник"]

    filtered_tokens = []

    stems = [stemmer.stem(t) for t in tokens]

    return stems


stopwords = nltk.corpus.stopwords.words("russian", "english")

# можно расширить список стоп-слов

stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на'])

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_featur = 200000

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000,

                                   min_df=0.01, stop_words=stopwords,

                                   use_idf=True, tokenizer=token_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

num_clusters = cnt_claster

# Метод к-средних - KMeans

from sklearn.cluster import KMeans

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

idx = km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

clusterkm = km.labels_.tolist()

frame = pd.DataFrame(texts)

frame

clusterkm = km.labels_.tolist()

frame = pd.DataFrame(texts)

# k-means

#out = {'text': texts, 'cluster': clusterkm, 'topic': df_res['mark']}

#frame1 = pd.DataFrame(out, columns=['text', 'cluster', 'mark'])

tech_text = "A4 X7-500MP Black"

text_vec = tfidf_vectorizer.transform([tech_text])

pd.DataFrame.sparse.from_spmatrix(text_vec)

km.predict(text_vec)