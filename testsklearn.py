from pprint import pprint

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

file = 'markandnom.csv'
data = pd.read_csv(file, delimiter=';', index_col=False, encoding='utf8')
data = data.replace(np.nan, '')

train_texts = data['nom'].tolist()
train_labels = data['mark'].tolist()

model = Word2Vec([train_texts,train_labels], min_count=1)
vect = model.wv.most_similar('Телевизор Manya')
pprint(vect)