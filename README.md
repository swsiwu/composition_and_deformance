# Composition and Deformance: Measuring Imageability with a Text-to-Image Model 

Paper link: [https://arxiv.org/abs/2306.03168](https://aclanthology.org/2023.wnu-1.16/)

Authors: Si Wu, David A. Smith

Published at the Workshop on Narrative Understanding at ACL 2023.

## Environment requirement
This is only tested with python 3.9:

```pip install -r requirements.txt```

Feel free to reach out if you encounter any problems setting up the environment!

## Datasets
**Conceptual 12M (CC12M)** (Changpinyo et al., 2021): available to download at https://github.com/google-research-datasets/conceptual-12m

**Cornell Newsroom Dataset** (Grusky et al., 2018): available to download after accepting the data licensing terms https://lil.nlp.cornell.edu/newsroom/download/index.html


Once you get their dataset, run
```python ./dataset/conceptual_captions/sample_cc12m_captions.py``` 
and
```./dataset/cornell_newsroom/sample_cornell_sentences.py```
to generate samples

## Run DALLE-mini for different deformances
```original.py permuted.py same_img.py backward.py just_noun.py``` are the scripts to run DALLE on input prompts, it will generate images and output their average CLIP score (aveCLIP). These files are here as a __reference__, you can __reuse the functions in those files to perform a deformance and use DALLE mini to generate images__. 


Notice that we organize our input and output in this following structure, 
```
caption_file = './dataset/cornell_newsroom/newsroom_samples.txt'
img_output_dir = './img_result/cornell_newsroom/same_img_noun/' 
output_text_dir = './dataset/cornell_newsroom/output_text/same_img_noun/'
poem_stats_path = './stats/cornell_newsroom/same_img_noun/'  + 'clip_stats' + '.csv'
```
input prompts are in ```caption_file = './dataset/cornell_newsroom/newsroom_samples.txt'``` where each line is a prompt,

for each input prompt, the generated image will be in ```img_output_dir = './img_result/[DATASET_NAME]/[DEFORMANCE_TYPE]/[PROMPT_ID]' ```,

then we also save a copy of the deformed text to ```output_text_dir = './dataset/cornell_newsroom/output_text/same_img_noun/'```,

finally, we save all the CLIP score information in ```poem_stats_path = './stats/cornell_newsroom/same_img_noun/'  + 'clip_stats' + '.csv'```



## Get image embeddings, imgSim, and aveCLIP
```python analyze_img.py``` to get image similarity score (imgSim) among the DALLE generated images of the same prompt. 


## To cite our paper
```
@inproceedings{wu-smith-2023-composition,
    title = "Composition and Deformance: Measuring Imageability with a Text-to-Image Model",
    author = "Wu, Si  and
      Smith, David",
    editor = "Akoury, Nader  and
      Clark, Elizabeth  and
      Iyyer, Mohit  and
      Chaturvedi, Snigdha  and
      Brahman, Faeze  and
      Chandu, Khyathi",
    booktitle = "Proceedings of the 5th Workshop on Narrative Understanding",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wnu-1.16",
    doi = "10.18653/v1/2023.wnu-1.16",
    pages = "106--117",
    abstract = "Although psycholinguists and psychologists have long studied the tendency of linguistic strings to evoke mental images in hearers or readers, most computational studies have applied this concept of imageability only to isolated words. Using recent developments in text-to-image generation models, such as DALLE mini, we propose computational methods that use generated images to measure the imageability of both single English words and connected text. We sample text prompts for image generation from three corpora: human-generated image captions, news article sentences, and poem lines. We subject these prompts to different deformances to examine the model{'}s ability to detect changes in imageability caused by compositional change. We find high correlation between the proposed computational measures of imageability and human judgments of individual words. We also find the proposed measures more consistently respond to changes in compositionality than baseline approaches. We discuss possible effects of model training and implications for the study of compositionality in text-to-image models.",
}
```
Please also feel free to reach out to us for questions and assistance!
