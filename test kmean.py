import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# Обучающие тексты и их ключевые слова

#train_texts = ["text1 with key1 key2", "text2 with key2 key3", "text3 with key1 key3"]
#train_labels = ["key1", "key2", "key1"]

filetext = Path('nom.txt')
lines = filetext.read_text(encoding='utf8').splitlines()
train_texts = [x.lower() for x in lines]

filetext = Path('mark.txt')
lines = filetext.read_text(encoding='utf8').splitlines()
train_labels = [x.lower() for x in lines]


# Создание объекта TfidfVectorizer для получения матрицы TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)

# Использование алгоритма KMeans для кластеризации текстов
kmeans = KMeans(n_clusters=418)
kmeans.fit(X_train)

# Использование алгоритма MultinomialNB для классификации текстов
clf = MultinomialNB()
clf.fit(X_train, train_labels)
# pickle.dump(kmeans, open("save_kmean_nom_model.pkl", "wb"))
# pickle.dump(clf, open("save_clf_nom_model.pkl", "wb"))

# Новый текст для классификации
new_text1 = ["Клавиатура Lenovo RTYGHFBEL653MS3"]
#new_text1 = ["Термометр для духовки ТБД в блистере", "Термометр для холодильника Айсберг ТБ-225"]
new_text = [x.lower() for x in new_text1]
#new_text =new_text1.lower()

# Получение матрицы TF-IDF для нового текста, на вход обязательно []
X_new = vectorizer.transform(new_text)

# Определение кластера для нового текста
pred_cluster = kmeans.predict(X_new)

# Определение ключевого слова для нового текста
pred_label = clf.predict(X_new)

print("Predicted cluster:", pred_cluster[0])
print("Predicted key word:", pred_label[0])

pred_proba = clf.predict_proba(X_new)
# Вывод вероятностей для каждого ключевого слова
for i, key_word in enumerate(clf.classes_):
    print("Probability of key word '{}': {:.2f}%".format(key_word, pred_proba[0][i]*100))

max_index = pred_proba[0].argmax()
pred_label = clf.classes_[max_index]

print("MAX Probability of key word '{}': {:.2f}%".format(pred_label, pred_proba[0][max_index]*100))