# Combination-of-Local-Models-for-Active-Learning
Grad Research Project for Advanced Machine Learning Course (Spring 19)

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
