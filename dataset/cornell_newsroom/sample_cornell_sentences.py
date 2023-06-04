import json
import gzip as gz
from nltk.tokenize import sent_tokenize
import random

path = "./data/train.jsonl.gz"
data = []

with gz.open(path) as f:
    for ln in f:
        obj = json.loads(ln)
        #data.append(obj)
        text = obj["text"]
        sentences = sent_tokenize(text)
        for sen in sentences:
            sen = sen.rstrip("\n").lstrip("\n").strip("\n").replace("\n", "")
            sen_len = len(sen.split(' '))
            if sen_len > 10 and sen_len < 30 and sen[0].isnumeric() == False and "|" not in sen and "Click" not in sen:
                if sen[0] != "$" and sen[0] != "(" and sen[0] != "@":
                    for letter in sen:
                        if letter.isalpha():
                            data.append(sen)
                            break
                
                
number_of_samples = 5000
random.shuffle(data)          
selected_sentences = data[:number_of_samples]

with open('newsroom_samples.txt', 'w') as f:
    for line in selected_sentences:
        f.write(line)
        f.write('\n')