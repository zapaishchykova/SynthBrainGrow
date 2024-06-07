# SynthBrainGrow: Synthetic Diffusion Brain Aging for Longitudinal MRI Data Generation in Young People
This directory for the code for the paper "SynthBrainGrow: Synthetic Diffusion Brain Aging for Longitudinal MRI Data Generation in Young People" [Arxiv](https://arxiv.org/pdf/2405.00682)

## Requirements
To run the example, you will need to install the [MONAI Generative Models package](https://github.com/Project-MONAI/GenerativeModels), [matplotlib](https://matplotlib.org/) and [Jupyter Notebook](https://jupyter.org/).
```pip install monai-generative```

## Data
Since [ABCD dataset](https://nda.nih.gov/abcd) needs to be formaly requested, we demonstrate the feasibility of the pipeline on the [OpenNeuro Dataset](https://openneuro.org/datasets/ds003604/versions/1.0.7) "A longitudinal neuroimaging dataset on language processing in children ages 5, 7, and 9 years old". We will only use pairs of images that were recorded at 7-9 years old, since method was initially trained on 8-16 years old subjects from ABCD dataset. 

Note: since this dataset requires N4 debiasing, it will take a while to preprocess all.


## Running the example
*Prerequisites: conda, gpu*

To run the example, you will need to run the Jupyter Notebook. This can be done by running the following command:

```jupyter notebook```

And open inference.ipynb in the browser.

## Citation
If you use this code, please cite the following paper:
```
@article{zapaishchykova2024synthbraingrow,
  title={SynthBrainGrow: Synthetic Diffusion Brain Aging for Longitudinal MRI Data Generation in Young People},
  author={Zapaishchykova, Anna and Kann, Benjamin H and Tak, Divyanshu and Ye, Zezhong and Haas-Kogan, Daphne A and Aerts, Hugo JWL},
  journal={arXiv preprint arXiv:2405.00682},
  year={2024}
}
```

## Work in progress
This repository is a work in progress.
[x] Add example notebook for inference
[] Add example notebook for training
[] Create Docker container for easy deployment

## LICENSE: CC BY-NC 4.0
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA. Copyright (c) [2023] [Diffusion Deep Learning for Brain Age Prediction and Longitudinal Tracking in Children though Adulthood]