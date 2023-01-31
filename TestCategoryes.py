import datetime
from pathlib import Path
from pprint import pprint

from gensim import corpora
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

#print("\n".join(twenty_train.data[3].split("\n")[:10])) #data одержит статью
#print(twenty_train.target_names[twenty_train.target[3]])#target содержит ид структуры
#print("\n")
#pprint(twenty_train.target)
#print("\n")

filetext = Path('nom.txt')
train_data = filetext.read_text(encoding='utf8').splitlines()

filetext = Path('mark.txt')
targetList = filetext.read_text(encoding='utf8').splitlines()

#Разбиваем targetlist на targetnames (список имен) and target (индекс имен)

for



#for t in twenty_train.target[:10]:
#    print(twenty_train.target_names[t])
#now = datetime.datetime.now()
#Предобработка текста, токенизация и отфильтровывание стоп-слов включены в состав высоко уровневого компонента,
# который позволяет создать словарь характерных признаков и перевести документы в векторы признаков:

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)
#pprint(twenty_train.data)
#pprint(X_train_counts.shape)
#pprint(X_train_counts)
#print(count_vect.vocabulary_.get(u'algorithm'))

#Следующее уточнение меры tf — это снижение веса слова, которое появляется во многих документах в корпусе,
# и отсюда является менее информативным, чем те, которые используются только в небольшой части корпуса.
# Примером низко ифнормативных слов могут служить служебные слова, артикли, предлоги, союзы и т.п.

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
pprint(X_train_tf.shape)

#Эти два шага могут быть объединены и дадут тот же результат на выходе, но быстрее,
# что можно сделать с помощью пропуска излишней обработки.
# Для этого нужно использовать метод fit_transform(..), как показано ниже:
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#pprint(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['pixel', 'Bible']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
print(predicted)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
 #   print(predicted == twenty_test.target)

#text_clf = Pipeline([    ('vect', CountVectorizer()),    ('tfidf', TfidfTransformer()),    ('clf', MultinomialNB()),])

#text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
#pprint(text_clf)

end = datetime.datetime.now()
print(end)
print(end-now)
