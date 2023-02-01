import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB

kmeans = pickle.load(open("save_kmean_nom_model.pkl", "rb"))
clf = pickle.load(open("save_clf_nom_model.pkl", "rb"))
vectorizer = pickle.load(open("save_vectorizer_nom_model.pkl", "rb"))

# Новый текст для классификации
#new_text1 = ["Клавиатура Lenovo RTYGHFBEL653MS3"]
new_text1 = ["Telek LG", "holodilnik Manya", "Детский автомобиль YD-518-1 красный", "Зернодробилка Фермер-3 ИЗЭ-14"]
new_text = [x.lower() for x in new_text1]
#new_text =new_text1.lower()

# Получение матрицы TF-IDF для нового текста, на вход обязательно []

X_new = vectorizer.transform(new_text)

# Определение кластера для нового текста
pred_cluster = kmeans.predict(X_new)

# Определение ключевого слова для нового текста
pred_label = clf.predict(X_new)
pred_proba = clf.predict_proba(X_new)

# print(X_new.size)

i = 0
for i in range(len(new_text)):
    print("Predicted cluster:", pred_cluster[i])
    print("Predicted key word:", pred_label[i])
    max_index = pred_proba[i].argmax()
    pred_proba_label = clf.classes_[max_index]
    print("MAX Probability of key word '{}': {:.2f}%".format(pred_proba_label, pred_proba[i][max_index]*100))
#print("Predicted cluster:", pred_cluster[1])
#print("Predicted key word:", pred_label[1])


#
# # Вывод вероятностей для каждого ключевого слова
# for i, key_word in enumerate(clf.classes_):
#     print("Probability of key word '{}': {:.2f}%".format(key_word, pred_proba[0][i]*100))
#
# max_index = pred_proba[0].argmax()
# pred_label = clf.classes_[max_index]
#
#
