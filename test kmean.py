from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB

# Обучающие тексты и их ключевые слова

train_texts = ["text1 with key1 key2", "text2 with key2 key3", "text3 with key1 key3"]
train_labels = ["key1", "key2", "key1"]

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

# Новый текст для классификации
new_text1 = "Встраиваемая СВЧ Печь Bosch BEL653MS3"
new_text =new_text1.lower()

# Получение матрицы TF-IDF для нового текста
X_new = vectorizer.transform([new_text])

# Определение кластера для нового текста
pred_cluster = kmeans.predict(X_new)

# Определение ключевого слова для нового текста
pred_label = clf.predict(X_new)

print("Predicted cluster:", pred_cluster[0])
print("Predicted key word:", pred_label[0])