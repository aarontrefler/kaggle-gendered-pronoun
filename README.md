Kaggle's Gendered Pronoun Resolution Competition 
==============================
Aaron Trefler  
Project Duration: Part-time work from March 22nd to April 18th

References
------------
Predictive modeling approach is heavily based on the public Kaggle kernel [Taming the BERT - a baseline](https://www.kaggle.com/mateiionita/taming-the-bert-a-baseline) 

Introduction
------------
This project was created in order to compete in [Kaggle's Gendered Pronoun Resolution (Pair pronouns to their correct entities)](https://www.kaggle.com/c/gendered-pronoun-resolution/overview) contest.

Contest description:
```
Can you help end gender bias in pronoun resolution?

Pronoun resolution is part of coreference resolution, the task of pairing an expression to its referring entity. This is an important task for natural language understanding, and the resolution of ambiguous pronouns is a longstanding challenge.

Unfortunately, recent studies have suggested gender bias among state-of-the-art coreference resolvers. Google AI Language aims to improve gender-fairness in modeling by releasing the Gendered Ambiguous Pronouns (GAP) dataset, containing gender-balanced pronouns (50% of its examples containing feminine pronouns, and 50% containing masculine pronouns).

In this two-stage competition, Kagglers are challenged to build pronoun resolution systems that perform equally well regardless of pronoun gender. Stage two's final evaluation will use a new dataset following the same format. To encourage gender-fair modeling, the ratio of masculine to feminine examples in the official test data will not be known ahead of time.
```

Executing Project
------------
In order to run the project:
1. ensure this project's home directory is put on your `$PYTHONPATH`, by altering `PROJ_PATH` variable in `src:run_setup.sh`
2. alter the `proj_path` variable in `src:utils.py` to that of this project's directory on your machine
3. download the [BERT-Base, Uncased model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and ensure it is locatable at `models/uncased_L-12_H-768_A-12`
4. execute notebooks in numerical order (ensure `proj_path` variable at beginning of all notebooks is properly set)

Project assumes you are running Python 3, and have all necessary packages.

Data Files
------------
Data files used in this project were provided by Kaggle and Google's [GAP Coreference Dataset](https://github.com/google-research-datasets/gap-coreference). Specifically the following raw files are used:
- `gap-test.tsv`: training dataset
- `gap_validation.tsv`: validation dataset
- `gap-development.tsv`: test dataset (competition stage 1)
- `test_stage_2.tsv`: test dataset (competition stage 2)

Features Used
------------
Context dependent [BERT model](https://github.com/google-research/bert) word embeddings were extracted for each of the following targets:
1. entity to be matched
2. first pronoun candidate
3. second pronoun candidate. 

Each target was converted into a 768 dimensional feature vector. Thus in total 2,304 features were used for prediction

Modeling
------------
Modeling was performed using a feed-forward neural network architecture, implemented with Keras using a TensorFlow back-end.

Jupyter (iPython) Notebooks
------------
Notebooks in this project are as follows:
- `AT-0.1-data-processing`: Performs data processing, and creates interim and clean datasets
- `AT-0.2-data-analysis`: Performs analysis on raw, interim, and clean datasets
- `AT-0.3-modeling`: Perform machine learning modeling

Project Organization
------------
    ├── README.md
    ├── bert                           <- Google BERT repo copied from GitHub
    │ 
    ├── data
    │   ├── clean                      <- Feature datasets
    │   ├── interim                    <- Processed datasets
    │   └── raw                        <- Data files downloaded from GAP dataset and Kaggle
    │
    ├── models                          
    │   ├── uncased_L-12_H-768_A-12    <- Pre-trained BERT model (must be downloaded)
    │   └── submissions                <- Kaggle submission file
    │
    ├── src
    │   ├── run_setup.sh               <- Setup environment
    │   ├── utils.sh                   <- Project level utility functions
    │   ├── data                       
    │   │   └── data_utils.py
    │   │  
    │   └── models                    
    │       ├── bert_model_utilsp.py
    │       └── keras_model_utils.py         
    │
    └── notebooks                      <- Jupyter notebooks

