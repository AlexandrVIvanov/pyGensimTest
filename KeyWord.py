from pathlib import Path
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB

#train_text = ["text1 with key1 key 2", "text2 with key2 key 3", "text3 with key1 key3"]
#train_label = ["key1", "key2", "key3"]

filetext = Path('nom.txt')
lines = filetext.read_text(encoding='utf8').splitlines()
train_text = [x.lower() for x in lines]

filetext = Path('mark.txt')
lines = filetext.read_text(encoding='utf8').splitlines()
train_label = [x.lower() for x in lines]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_text)

vectlabel = TfidfVectorizer()
X_label = vectlabel.transform(train_label)

kmeans = KMeans(n_clusters=418)
kmeans.fit(X_train)

clf = MultinomialNB()
clf.fit(X_train, train_label)

new_text = "samsung"
X_new = vectorizer.transform([new_text])

pred_cluster = kmeans.predict(X_new)

pprint(train_label[pred_cluster[0]])



