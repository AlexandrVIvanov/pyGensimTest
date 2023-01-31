import pandas as pd

filetext = Path('nom.txt')
lines = filetext.read_text(encoding='utf8').splitlines()
train_text = [x.lower() for x in lines]

filetext = Path('mark.txt')
lines = filetext.read_text(encoding='utf8').splitlines()
topic_text = [x.lower() for x in lines]

data = pd.read_csv()


