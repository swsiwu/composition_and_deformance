import os
import time
import random
import random
from dalle_mini import DalleBartProcessor
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
from flax.training.common_utils import shard
import jax
import jax.numpy as jnp

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

from flax.jax_utils import replicate
from functools import partial
import re
import csv
import warnings
import gzip
import pickle
import pandas as pd
import shutil
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


warnings.filterwarnings("ignore")

"""
https://github.com/borisdayma/dalle-mini/blob/2f1e5d9780c606d446a49cbd0d368fc0fee67959/tools/inference/inference_pipeline.ipynb

"""
# Model references

# dalle-mega
#DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or Hub or local folder or google bucket
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# CLIP model
CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_COMMIT_ID = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"



"""

"""


# check how many devices are available
jax.local_device_count()

"""

"""
# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)



from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)


# Load CLIP
clip, clip_params = FlaxCLIPModel.from_pretrained(
    CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
clip_params = replicate(clip_params)



# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)


@partial(jax.pmap, axis_name="batch")
def p_clip(inputs, params):
    logits = clip(params=params, **inputs).logits_per_image
    return logits

"""

"""

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

# number of predictions
n_predictions = 16

# We can customize top_k/top_p used for generating samples
gen_top_k = None
gen_top_p = None
temperature = 0.85
cond_scale = 3.0
# create a random key
    
def get_ave_clip_score(prompt, line_out_img_path):
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    
    tokenized_prompt = processor([prompt])
    tokenized_prompt = replicate(tokenized_prompt)


    # generate images
    images = []
    clip_scores = []
    
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for img in decoded_images:
            images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))
    
    # get clip scores
    clip_inputs = clip_processor(
        text=[prompt] * jax.device_count(),
        images=images,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).data
    logits = p_clip(shard(clip_inputs), clip_params)
    logits = logits.squeeze().flatten()


    counter = 0
    whatsprinted = '[START]' + str(prompt)
    print(whatsprinted)
    for idx in logits.argsort()[::-1]:
        counter += 1
        if counter <= 16:
            out_file_name = line_out_img_path + str("{:.2f}".format(logits[idx])) +  '.png'
            images[idx].save(out_file_name)
            clip_scores.append("{:.2f}".format(logits[idx]))
            #print(f"Score: {logits[idx]:.2f}\n")

        else:
            break
            
    return clip_scores





def get_just_noun(line):
    tokens = nltk.word_tokenize(line)
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    just_noun_line = " ".join(nouns)
    #print(just_noun_line)
    return just_noun_line

    
    
def generate_poem_img(lines, dalle_img_output_dir):
    counter = 0
    prompt = ''
    clip_scores = []
    ave_clip_scores = 0.0
    ave_denominator = 0.0
    edited_lines = []
    
    for line in lines:
        counter += 1
        if counter % 2 == 0:
            just_nouns_line = get_just_noun(line)
            prompt = prompt + ' ' + just_nouns_line
            
            line_out_img_path = dalle_img_output_dir + str(counter) + '/'
            
            if not os.path.exists(line_out_img_path):
                os.makedirs(line_out_img_path)
            
            edited_line = prompt 
            edited_lines.append(edited_line)
            line_clip_scores = get_ave_clip_score(prompt, line_out_img_path)
            clip_scores.append(line_clip_scores[0])
            ave_clip_scores += float(line_clip_scores[0])
            ave_denominator += 1.0
            
            prompt = ''
        else:
            ### reverse:
            just_nouns_line = get_just_noun(line)
            prompt = just_nouns_line
    if prompt != '':
        #counter += 1
        line_out_img_path = dalle_img_output_dir + str(counter) + '/'
        if not os.path.exists(line_out_img_path):
                os.makedirs(line_out_img_path)

        edited_line = prompt 
        edited_lines.append(edited_line)
        line_clip_scores = get_ave_clip_score(prompt, line_out_img_path)
        clip_scores.append(line_clip_scores[0])
        ave_clip_scores += float(line_clip_scores[0])
        ave_denominator += 1.0

        prompt = ''
    if ave_denominator == 0:
        ave_denominator = 1.0
            
    return clip_scores, float(ave_clip_scores/ave_denominator), edited_lines
    
def output_poem(output_dir, filename, lines):
    outfile_path = output_dir + filename + '.txt'
    with open(outfile_path, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

# no noun at all: I also get nauseous and feel like I can't eat when it is really bad, but I force myself to stay healthy," she said.
caption_file = './dataset/cornell_newsroom/newsroom_samples.txt'
img_output_dir = './img_result/cornell_newsroom/just_noun/' 
output_text_dir = './dataset/cornell_newsroom/output_text/just_noun/'
poem_stats_path = './stats/cornell_newsroom/just_noun/'  + 'clip_stats' + '.csv'
stats_file = open(poem_stats_path, 'w')

with open(caption_file, newline='\n') as txtfile:
    caption_lines = txtfile.readlines()
    counter = 0
    for line in caption_lines:
        line = line.rstrip('\n')

        lines = []
        lines.append(line)
        #output_poem(output_text_dir, str(counter), lines)
        dalle_img_output_dir = img_output_dir + str(counter) + '/'
        if not os.path.exists(dalle_img_output_dir):
            os.makedirs(dalle_img_output_dir)
        clip_scores, ave_clip, edited_lines = generate_poem_img(lines, dalle_img_output_dir)
        output_poem(output_text_dir, str(counter), edited_lines)
        stats_print = str(counter) + ',' + str(ave_clip) + ',' +  re.sub(',',' ',str(clip_scores)).strip('[]')
        stats_file.write(stats_print)
        stats_file.write('\n')
        
        
stats_file.close()