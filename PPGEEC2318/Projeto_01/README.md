# Model Card
Model cards are a succinct approach for documenting the creation, use, and shortcomings of a model. The idea is to write a documentation such that a non-expert can understand the model card's contents. For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Ivanovitch Silva and João Gabriel Costa Rodrigues created the model. A pipeline was built using Pytorch and Scikit-Learn to train a Logistic Regression model. For the sake of understanding, some simples hyperparameter-tuning was conducted, and the hyperparameters values adopted in the train are described in a [yaml file](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_01/params.yaml)
* Model date: 11/05/2025
* Model version:v.01

## Intended Use
This model is used as a proof of concept for the evaluation of an entire data pipeline incorporating MLOps assumptions. The data pipeline is composed of the following stages: a) ``data``, b) ``eda``, c) ``preprocess``, d) ``check data``, e) ``segregate``, f) ``train``, g) ``evaluate``.

## Training Data
This dataset contains information on customer purchase behavior across various attributes, aiming to help data scientists and analysts understand the factors influencing purchase decisions. The dataset includes demographic information, purchasing habits, and other relevant features.
You can download the data from the [kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset)
It was noted that the training data is imbalanced when considered the target variable (PurchaseStatus) and some features.
The continuous quantitative data features have been grouped to improve network performance (learning) by avoiding overfitting and underfitting. The “Age” feature will also be grouped.

Describe: ![image](https://github.com/user-attachments/assets/39afbf26-06f3-4f4b-8e37-e367d520b6c9)


![image](https://github.com/user-attachments/assets/8c4afdf5-a4f1-49ef-a562-df0cd02f11bf)


## Evaluation Data
The dataset under study is split into Train and Test during the ``Segregate``. 70% of the clean data is used to Train and the remaining 30% to Test.  This configuration is done in a [yaml file](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_01/params.yaml).


The loss graph by epochs shows better model performance with validation data and no signs of overfitting
![image](https://github.com/user-attachments/assets/afabe384-c4a6-45ae-ad7d-805ac92e515c)


## Metrics
In order to follow the performance of machine learning experiments, the project utilities accuracy, F1, precision, recall as metrics.The metrics adopted they are: [classification report](https://scikitlearn.org/stable/modules/generated/sklearn.metrics.classification_report.html), [confusion matrix](https://scikitlearn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

![image](https://github.com/user-attachments/assets/a7673c72-9158-41f6-99ef-51939a74ba0c)

![image](https://github.com/user-attachments/assets/5a61d34f-7413-44cc-bcd6-aa945343b8bc)



## Ethical Considerations
We may be tempted to claim that this dataset contains the some attributes capable of predicting factores of buy someone's but the metrics must be improved. the accuracy obtained was 83%

## Caveats and Recommendations
It should be noted that the model trained in this project was used only for validation of a complete data pipeline. It is notary that some important issues related to dataset imbalances exist, and adequate techniques need to be adopted in order to balance it.
