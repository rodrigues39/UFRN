# Model Card
Model cards are a succinct approach for documenting the creation, use, and shortcomings of a model. The idea is to write a documentation such that a non-expert can understand the model card's contents. For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A complete training pipeline was built using **PyTorch** and a set of modular **Convolutional Neural Network (CNN)** architectures, specifically configured for the classification of sports images.

All architectures and hyperparameters are defined in a configurable [YAML file](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/params.yaml), enabling reproducible experimentation and simplified model switching.

The core of the pipeline is a general-purpose training class that acts as an end-to-end trainer, encapsulating:
- Training and validation routines
- Data preprocessing and augmentation
- Optimizer, scheduler, and loss configuration
- Evaluation and visualization of metrics
- FindingLR

- **Model date:** 16/07/2025  
- **Model names:** `model_cnn`, `model_cnn_nf`, `model_cnn_b`, `model_cnn_ex`  
- **Architecture types:** Custom CNNs (`CNN2`, `CNN2_M2`, `CNN2_M3`, `CNN2_M4`)  
- **Framework:** PyTorch  
- **Input size:** 224×224 RGB  
- **Total parameters:** Varies depending on architecture  
- **Authors:** Ivanovitch Silva and João Gabriel Costa Rodrigues  
- **Version:** 1.0

## Intended Use
- **Task:** Multiclass image classification of 10 distinct sports categories
- **Audience:** Researchers, ML students, computer vision engineers
- **Intended for:** Educational use, architecture benchmarking, overfitting analysis, and data augmentation studies
- **Not intended for:** Production or critical decision-making without external validation or real-world testing

## Factors
Several factors that can impact model performance were considered and addressed during dataset preparation and model training, including class imbalance, visual variability, data augmentation, and image standardization. Additionally, model layer activations and gradients were monitored using hook functions, enabling deeper analysis and debugging during training.

## Monitoring & Debugging
Hook functions were used during training to monitor activations and gradients from intermediate layers. This enabled real-time inspection of feature maps and gradient flow, supporting better debugging and architectural adjustments.
This is an example of a [hook](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/hook_cnn.png), capturing the activations from the first convolutional layer of the CNN model

## Metrics
To effectively monitor and compare the performance of machine learning experiments, this project adopts the following evaluation metrics:
**Accuracy**, **Precision**, **Recall** and **F1-Score**. Using:  [classification report](https://scikitlearn.org/stable/modules/generated/sklearn.metrics.classification_report.html), [confusion matrix](https://scikitlearn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
### Classification Reports
[`model_cnn`](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/report_cnn.jpg)
[`model_cnn_nf`](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/report_cnn_nf.jpg)
[`model_cnn_b`](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/report_cnn_b.jpg)
[`model_cnn_ex`](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/report_cnn_ex.jpg)

### Confusion Matrix
![image](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/matriz%20confusion02.png)

## Evaluation Data
The loss graph by epochs shows better model performance with validation data and no signs of overfitting
[loss_model_cnn](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/loss_01_lr.png), 
[loss_model_cnn_nf](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/loss_02_lr.png), 
[loss_model_cnn_b](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/loss_03_lr.png) and
[loss_model_cnn_ex](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/loss_04_lr.png).

## Finding

## Training Data
Classes were chosen for their visual diversity and representativeness. The goal was to create a balanced and interpretable benchmark for evaluating CNN architectures.
- **Source:** Same dataset used for evaluation — `10_Sports_Classes`, filtered from the original Kaggle dataset.
- **Classes:** baseball, basketball, bmx, football, formula 1 racing, nascar racing, rugby, tennis, track bicycle, volleyball
- **Split:**  
  - Train: 1,205 images  
  - Validation: 259 images  
  - Test: 263 images  
  The same split is maintained across all experiments for consistency.

- **Data Augmentation:**  
  Applied **only to the training set** to improve generalization:  
  - Random horizontal flip  
  - Random rotation (±10°)

- **Class Weights:**  
  Calculated from the training distribution and applied to the loss function to mitigate class imbalance.
  Values used: [0.9414, 0.9640, 1.1476,0.8607,0.8607, 0.8669, 0.9797, 1.2296, 1.1931,1.1368]

## Quantitative Analyses
Despite architectural differences between the models, overall performance remained consistently high. The 'Novo Modelo' variant demonstrated superior results, especially in handling complex classes.

- **Final Validation Accuracy:**  
  Accuracy scores were close across all models, with minor variations, indicating that model design alone was not the primary limiting factor.

- **Classification Report:**  
  F1-scores for most models ranged between **0.58 and 0.64**, with accuracy for some individual classes varying from **54% to 61%**, particularly in visually similar categories.

These findings suggest that, beyond model improvements, performance is also heavily influenced by **image quality, diversity**, and **intra-class visual similarity**, which remain key challenges in this classification task.
![image](https://github.com/rodrigues39/UFRN/blob/main/PPGEEC2318/Projeto_02/data/matriz%20confusion.png)


## Ethical Considerations
The dataset used in this project consists exclusively of sports images collected from public sources and does not include sensitive personal data such as faces, demographic attributes (e.g., race, gender), or private content.

No explicit demographic bias was identified in the selected images, as the focus is on sports equipment, scenes, and activities rather than individuals.

However, it is important to note that this model is specifically trained for **sports classification**. Applying it in domains outside of sports imagery may lead to unintended biases or misclassifications, as the training data does not generalize to broader visual contexts.

## Caveats and Recommendations
Although the model performs well on the curated sports image dataset, its performance may degrade significantly when applied to images from different domains — such as low-resolution smartphone photos, video frames, or non-sports contexts.
If the target dataset differs substantially in style, quality, or content, it is strongly recommended to **retrain the model or recalibrate class weights** using the new data distribution.
Before deploying the model in production or new applications, a **critical evaluation in the target environment** should be conducted to ensure robustness, fairness, and reliability.
