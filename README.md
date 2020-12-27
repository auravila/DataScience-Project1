# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**


This dataset contains results of marketing contact campaings from a set of bank customers. This dataset includes customer demographics, consumer & confidence indexes and other statistics. 

We seek to predict the likelihood of contracting a loan after being targeted and contacted via a marketing campaign (y).  


**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The approach taken to create the experiments was based on a classification model where the random sampling mode was utilized. The accuracy results of 0.90 provides great confidence in the experiment run nevertheless for this excercise it is just based in the result of a discrete set of values and with a small set of iteractions. 

Random Sampling was used over Grid search mainly due to grid search evaluates all of the combinations of parameters and incurs in a large amounts of time and computational resources whilst random search consists in sampling random values in the hyperparameter space.


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The pipeline is a sequential call to a train script which references a dataset, cleanses it, splits the data into a train and test sets and calls a logistics regresion model.
The Logistics regresion model parameters are provided by the hyperdrive model Random sampling and the discrete set of parameters ared defined by choice.

The Logistics regresion for this scenario takes two argumes as input: C and Max Iterations. There parameters are mapped to the RandomSapling functiond defined by choice.

C : float, optional (default=1.0)
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

C is used in order to get a perfect trade off between bias and high variance in the model. C is used to maintaing the following relationship: Lambda = 1/C

max_iter : int
Defines the maximum number of iterations taken for the model to converge. Default value = 100 but this can be changed depending on the volume of the train data.

Reference:
https://scikit-learn.org/0.16/modules/generated/sklearn.linear_model.LogisticRegression.html


The primery metric selected for an optimized return was accuracy which is constrained by a bandit policy, i.e. if a particular run does not meet the threshold this child run is terminated.

The Pipeline:
1-) Load data from a dataset
2-) Clean data (train.py)
3-) Split data (train.py)
4-) Train data (train.py)
5-) Assign Logistic Regresion to the model (train.py)
6-) Configure Hyperparameters (primary metric, parameters sample model)
7-) Define Bandit Policy
8-) Run Experiment (Loop until optimized metric is returned or bandit policy met )

Draft Example of how the architecture may look like:

![alt text](https://github.com/auravila/DataScience-Project1/blob/main/Experiment%20Pipeline%20Example.png)


**What are the benefits of the parameter sampler you chose?**

Choosing a discrete set of parameters guarantees the same probability for each value of the list. A small set of parameters provides a quick turnaround of the
experiment results for initial evaluation.

Random Sampling is a faster evaluation model and does not requires extensional compute resources as opposed to grid search

**What are the benefits of the early stopping policy you chose?**

The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.

The Bandit policy takes the following parameters:

slack_factor or slack_amount: The slack allowed with respect to the best performing training run. slack_factor specifies the allowable slack as a ration. slack_amount specifies the allowable slack as an absolute amount, instead of a ratio.

evaluation_interval: Optional. The frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

delay_evaluation: Optional. The number of intervals to delay the policy evaluation. Use this parameter to avoid premature termination of training runs. If specified, the policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation.

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

Reference: https://rdrr.io/github/Azure/azureml-sdk-for-r/man/bandit_policy.html

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML differs from hyperdrive mainly because AutoML runs multipe pipelines in paralled with different algorithms and parameters at the same time. Hyperdrive runs parameters are limited by time and cost.

The model:

1-) Load data from a dataset
2-) Train this data (random split)
3-) Configure AutoMl parameters (classification model,timeout,primary_metric)
4-) Create Experiment (Assing AutoML Config)
5-) Run Experiment

For this classification excecise the model that produced the best outcome was the VotingEnsemble and compared to the hyperparameter runs the AutoML produced a better result with a 1% greater accuracy outcome.  

The idea behind the VotingClassifier is to combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels. Such a classifier can be useful for a set of equally well performing model in order to balance out their individual weaknesses. 

Reference: https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier

List of Hyperparameters values best_run.get_tags()

{'_aml_system_azureml.automlComponent': 'AutoML',
 '_aml_system_ComputeTargetStatus': '{"AllocationState":"steady","PreparingNodeCount":0,"RunningNodeCount":1,"CurrentNodeCount":1}',
 'ensembled_iterations': '[1, 19, 0, 22, 14, 20, 4, 5]',
 'ensembled_algorithms': "['XGBoostClassifier', 'LightGBM', 'LightGBM', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'RandomForest', 'RandomForest']",
 'ensemble_weights': '[0.21428571428571427, 0.21428571428571427, 0.14285714285714285, 0.14285714285714285, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142]',
 'best_individual_pipeline_score': '0.9150270346813996',
 'best_individual_iteration': '1',
 '_aml_system_automl_is_child_run_end_telemetry_event_logged': 'True',
 'model_explain_run_id': 'AutoML_9e426425-1c45-4632-b78c-52d20e604402_ModelExplain',
 'model_explanation': 'True'}

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

AutoML model is more accurate than the random sampling model by 1%. AutoML also calculated all of the additional metrics and selected the most optimized combination
of metris by the VotingEnsemble algorithm. Automl provided a holistic result for each of the metrics reassuring that the model is accurate enough for prediction.

Regarding architecture difference, I believe is mainly the sequential execution of a defined experiment by multiple child runs as opposed to the AutoML execution 
which runs through all of the available algorithms and retrieves the best and most optimized resultset.

For this particular scenario 1% difference is inmaterial but my best guess here is that the difference may be due to the poor selection of hyperdrive parameters of the model and also the different approaches taken to train the datasets. e.g. For the first experiment train_test_split was used and for AutoML random_split.

AzureML simplifies modelling but it could limits knowledgeable developers to use and customize their own algorithms.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

Try out other types of model and algorithms with a larger sets of parameters, use the same tranining mechanisms. Generate simple predictions to confirm the validity of the model. 

Also try out a different metrics within the chosen model such as:

•	Precision: It answers the question: When the classifier predicts yes, how often is it correc

•	Recall: It answers the question: When it’s actually Yes, how often does the classifier predict yes?

•	False Positive Rate (FPR) : It answers the question: When it’s actually no, how often does the classifier predict Yes?

•	F1 Score: This is a harmonic mean of the Recall and Precision. Mathematically calculated as (2 x precision x recall)/(precision+recall).

Other options:

Run models with the grid search and the Bayesion Parameter sampling. 

A Bayesian optimizer samples a subset of hyperparameters combinations and the difference between random sampling and grid search resides in the way each combination is chosen.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

Image attached to main project - root folders

Command used
delcluster = ComputeTarget.delete(MYcompute_cluster)

![alt text](https://github.com/auravila/DataScience-Project1/blob/main/Cluster%20Delete.jpeg)

