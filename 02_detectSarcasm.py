# Import libraries and modules
import json
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
data = pd.DataFrame([json.loads(line)
                     for line in open('G:/My Projects/tfNLP/_data/Sarcasm_Headlines_Dataset.json', 'r')])
print(data.info())

# tokenizer operations
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(data['headline'])

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)

sequences = tokenizer.texts_to_sequences(data['headline'])
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)