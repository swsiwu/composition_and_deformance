import os
import random


"""

using Google 12M conceptual captions dataset

https://github.com/google-research-datasets/conceptual-12m

"""


number_of_samples = 5000

captions = []
selected_captions = []

with open('./cc12m.tsv','r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\t')
        caption = line[1]
        caption = caption.strip('\n')
        caption = caption.lstrip()
        if "<" not in caption and "#" not in caption:
            captions.append(caption)
            
random.shuffle(captions)          
selected_captions = captions[:number_of_samples]

with open('caption_samples.txt', 'w') as f:
    for line in selected_captions:
        f.write(line)
        f.write('\n')
    