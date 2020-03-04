# Combination-of-Local-Models-for-Active-Learning
Grad Research Project for Advanced Machine Learning Course (Spring 19)

Disclaimer – The code for this project in Github is a non-edited baseline version that was not implemented in an object-oriented style and we have omitted lots of pre-processing steps and modeling for privacy purposes since the algorithm is still under work. The version was only for submission of a graduate project to be implemented in 2 months which later turned to a research project. 

In this project we introduce a novel Active Learning measure of uncertainty approach based on multidimensional minimum distance. Multiple local models utilize this measure to query an oracle in a classification task. Our experiments show good accuracy results when compared to an existing active learning framework working on the same classification dataset. We discuss the number of initial labeled samples, queries and features and how each of these parameters factored into our final results.

Algorithm:

1. Select an equal number of positive and negative class samples from the initial labeled samples.
2. Select the desired number of features to use.
3. Calculate distance between opposite labelled instances in our labelled subset.
4. Select opposite points with the shortest distance between them and calculate their midpoint.
5. For every midpoint point choose KONN instance points.
6. Generate local models for classification.
7. Select uncertain samples and query oracle.
8. Feed remaining unlabeled samples to the local models until all instances are labeled.

Experiments    
 
In this section we’ll explain how we used the breast cancer dataset to run experiments for the implementation of our algorithm and how our results compared to results obtained from using the same dataset on the modAL active learning framework. Our main focus will be in how we varied the following parameters during experimentation:  
• Initial size of labeled samples (L). 
• Number of k-opposite nearest neighbors (k). 
• Number of features. 
• Number of queries to oracle (Q). 
 
1. Initial size of labeled samples (L): the breast cancer dataset has a total of 569 instances. From this total we selected 20, 60, and 100 total labeled samples with a split of 50% of one class and 50% of another class. 
 
2. Number of k-opposite nearest neighbors (k): for this parameter we kept the k=4 for all of our experiments. 
 
3. Number of features: our features parameters were tested using 2, 5, 7, 10 and even 20 features for a dataset that has a total of 30 features to choose from. All of these features were selected used recursive feature elimination (RFE). 
 
4. Number of queries to oracle (Q): The number of queries that our algorithm asked the oracle varied between 0, 5, 10, 15, and 20. We stopped at a twenty query maximum to respect the general active learning assumption that labeled samples are costly or difficult to obtain.


1. Data Collection 
We chose datasets from public repositories like Kaggle and  UCI ML repository and the datasets that were chosen in many of the Active Learning research papers to have an Apple-to-Apple comparison. Distribution of domains in those chosen datasets was finance, health-care, automobile, technology, banking, etc which are pretty diverse in order to check the adaptability of the algorithm that we had developed. Most of the data was in file formats .txt, .csv, .json, .dat. We included specific functionality to handle all the formats. 

2. Data Processing
 Most of the datasets had missing or corrupted values and even outliers, so mostly we utilized the Robust Scalers instead of standard. In some cases the data was sparse so we went with MaxAbsScaler(). In order to fight skewness, we used the power transformation method. In the case of handling the missing data, we developed pipelines of 3 different methods: 1) mean 2)median 3) KNearestNeighbors to see the pattern of working for different domains. Since we mostly dealt with high cardinality categorical variables we used the count-based encoding or target encoding. 

3. Model Building
For this step, we had to randomly sample some percentage of training(labeled data) to initialize our algorithm. Here we picked the samples through stratification on labels to have a balanced dataset. After picking the training samples we use the rest unpicked samples and their true labels as our test set.
We had to train our local models with this small percentage of training data, but the good thing is we have many localized models, hyperparameters to tune where the percentage of training data to be sampled and a number of local learners). Due to variation in our preprocessing steps, we have 3 different pre-processed data to be fed to our model. For now, we did not do much feature selection since we had no domain knowledge on many of the datasets. Now we use our model to get the labels of our test data

4. Model Evaluation
We now compared the labels and also grouped the dataset to check the accuracy for each class in order to figure out bias. We performed hyperparameter tuning to see the change in the Model's performance. Finally, we also ran the whole dataset using a tree-based model to get an overview of feature importances, so that we can go back and remove some irrelevant features. We used a grid search to reach a good parameter setting for the model to perform well on average across all datasets.

Future work
Currently, we are working to improve our sampling of labelled instances and adjusting our local classifier's learning procedure to better understand the dataset. We are also working with local SVM's instead of logistic regression classifiers to see its performance across various datasets. We are also working on a parameter that decides the number of local classifiers.

