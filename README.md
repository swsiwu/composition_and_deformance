# Composition and Deformance: Measuring Imageability with a Text-to-Image Model 

paper link: https://arxiv.org/abs/2306.03168

To be appear at the Workshop on Narrative Understanding at ACL 2023!

## Environment requirement
We will upload a .yml file soon!

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
@misc{wu2023composition,
      title={Composition and Deformance: Measuring Imageability with a Text-to-Image Model}, 
      author={Si Wu and David A. Smith},
      year={2023},
      eprint={2306.03168},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Please also feel free to reach out to us for questions and assistance!
