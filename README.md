# Composition and Deformance: Measuring Imageability with a Text-to-Image Model 

paper link: 

## Datasets
**Conceptual 12M (CC12M)** (Changpinyo et al., 2021): available to download at https://github.com/google-research-datasets/conceptual-12m

**Cornell Newsroom Dataset** (Grusky et al., 2018): available to download after accepting the data licensing terms https://lil.nlp.cornell.edu/newsroom/download/index.html


Once you get their dataset, run
```python ./dataset/conceptual_captions/sample_cc12m_captions.py``` 
and
```./dataset/cornell_newsroom/sample_cornell_sentences.py```
to generate samples

## Run DALLE-mini for different deformances
```original.py permuted.py same_img.py backward.py just_noun.py``` are the scripts to run DALLE on input prompts, it will generate images and output their average CLIP score (aveCLIP)


## Get image embeddings, imgSim, and aveCLIP
```python analyze_img.py``` to get image similarity score (imgSim) among the DALLE generated images of the same prompt. 


## To cite our paper
