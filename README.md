# Content-based Recommender System for Detecting Complementary Products

## Overview of the pipeline and experiments

### Introduction
This is the code for the experiments used in a master thesis research. In this part we will briefly explain the structure of the notebooks and python files implemented, and a description of the overall process. The following figure explains the process during the thesis work, starting from the Data Retrieval and ending with different Comparative Analysis.

<img src="https://github.com/marinaangelovska/complementary_products_suggestions/blob/master/report_structure.png" width="700">

Each of these steps is separated in different notebooks, where some of the repetitive functions are split in a few different .py files. We do not include everything which was done during this thesis in these files, but only the most important and relevant experiments and models are included. 

## Notebooks description:

### 1. Data retrieval and preprocessing & 2. Data Analysis
The first two original notebooks were removed because of data confidentiality. Instead, we use two fake datasets (named \"dataset\" and \"content\") with a few samples just so that we can explain what kind of data is needed for running the models. The \"dataset\" consists of pairs of matches of products with their identifiers, titles and the label indicating if the second product is an add-on to the first product or not. The \"content\" dataset contains all products with all of their attributes (in this dummy data, only the title).

### 3. CNN vs. LSTM &rarr; 03-CNN_vs._LSTM.ipnyb
In this notebook we do the siamese neural network implementation for CNN and LSTM. We introduce the differences when performing Intermediate and Late Merge in the two pipelines. At the end after training and testing each of the proposed networks, we do comparative analysis by plotting the accuracy and loss during training, and the final accuracy and AUC score for the predictions. An overview of a Siamese Neural Network is shown on the Figure below:

<img src="https://github.com/marinaangelovska/complementary_products_suggestions/blob/master/snn.png" width="300">

### 4. Word2vec embeddings &rarr; 04-LSTM_embeddings_tests.ipnyb
From this point, we only use the Siamese LSTM approach with Intermediate Merge as it was the model that gave the best resutls. We now want to see how will removing the Word2vec embeddings before the Embedding layer in the Siamese LSTM impact the performance of the model. For this purpose, we run two models in paralel, one with word2vec embeddings included and one model without (this means that the weights in the Embedding layer will be randomly initialized and it might not generalize well in the test set as the embeddings will be created only on the training set).


### 5. Siamese LSTM compared to Baselines &rarr; 05-baselines.ipnyb
In this part we implement and test the baselines on the same data we used on the Siamese LSTM before. Random Forest, Vanilla Neural Network and Single (not Siamese) LSTM are implemented. We report their performances using the same metrics.

### 6. Using the Siamese LSTM weights for transforming the solution to KNN (cosine similarity) problem &rarr; 06-LSTM_to_KNN.ipnyb
We are creating a list of target and candidate products and we want to run all possible pairs in the model so that for each target product we get K add-ons product. In real-life scenarios we would get millions of product pairs, thus we need to find a solution which will handle these data points in the most efficient way. Therefore, in this notebook we train Siamese LSTM on the training set and then we save the weights/embeddings for each product for the target and candidate products sets. After doing this, we already have the vector representations for each product. We then calculate the cosine similarity between all possible pairs of products from the two sets and we finally get their similarity/complementarity score. The difference with running all of these product pairs through the neural network is that we would have iterated through it for each pair, but with the approach propose we only iterate once for each product, as explained in the following picture.

<img src="https://github.com/marinaangelovska/complementary_products_suggestions/blob/master/snn_scalable.png" width="300">

## Python files description:

### 1. Data Retrieval
This files contains queries for getting the data from BigQuery, but for the reasons named before, we do not include this file in this repo.

### 2. Data Preprocessing &rarr; data_preprocessing.py
This file consisits helper functions for generating the negative samples, merging two dataframes and cleaning the data.

### 3. Embeddings &rarr; embeddings.py
Here we perform Word2vec embeddings.

### 4. Helper Functions &rarr; helper_functions.py
This file consists the train and test set tokenization as well as the train-test-split. We need these two functions for almost every experiment.

### 5. Config &rarr; config.py
All of the hyperparameters are included here.

## Getting started
To get started with the notebooks, open the terminal in the main folder of the project and run the following commands:
1. ```pip install poetry```
2. ```poetry config virtualenvs.in-project true```
3. ```poetry run``` to start jupyter lab in the virtual env
* Run `poetry add [package]` to add a python package.
