# Sports Image Classification – Neural Computing Project

**Author:** Giuseppe Lombardia   
**Environment:** MATLAB (Deep Learning Toolbox)

---

## 1. Overview
This repository contains the final project developed for the *Neural Computing* course.  
The goal was to implement and evaluate **transfer learning** models in **MATLAB** for the task of **sports image classification** across 100 categories.

All code snippets, results, and visual analyses are included within the full report:
📄 [`report/Neural_computing.pdf`](report/Neural_computing.pdf)

---

## 2. Objectives
- Apply **transfer learning** using pretrained CNNs (*GoogLeNet* and *ResNet-18*).  
- Preprocess and resize RGB images using `augmentedImageDatastore`.  
- Compare architectures in terms of accuracy, convergence speed, and generalization.  
- Evaluate models with **confusion matrices** and visual examples of predictions.

---

## 3. Models and Results

| Model | Framework | Validation Accuracy | Test Accuracy |
|--------|------------|--------------------|----------------|
| GoogLeNet | MATLAB Deep Learning Toolbox | 94.0 % | 96.4 % |
| ResNet-18 | MATLAB Deep Learning Toolbox | 89.4 % | 92.6 % |

Both architectures were fine-tuned on the *Sports Image Classification* dataset (100 classes, RGB 224×224×3).  
GoogLeNet achieved the best overall performance and generalization on unseen data.

---

## 4. Dataset
Dataset used: [Sports Image Classification – Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)

| Split | Images | Description |
|--------|--------|--------------|
| Training | 13,492 | Used for fine-tuning pretrained networks |
| Validation | 500 | Used for model selection |
| Test | 500 | Used for final evaluation |

All images were resized to **224×224 px** and standardized to **RGB format**.

---

## 5. MATLAB Code Example

Example of the data loading and evaluation workflow:

```matlab
imagesize = [224,224,3];
train = imageDatastore("train/","IncludeSubfolders",true,"LabelSource","foldernames");
valid = imageDatastore("valid/","IncludeSubfolders",true,"LabelSource","foldernames");
test  = imageDatastore("test/","IncludeSubfolders",true,"LabelSource","foldernames");

resizedTrain = augmentedImageDatastore(imagesize,train,"ColorPreprocessing","gray2rgb");
resizedValid = augmentedImageDatastore(imagesize,valid,"ColorPreprocessing","gray2rgb");
resizedTest  = augmentedImageDatastore(imagesize,test,"ColorPreprocessing","gray2rgb");

pred_test_labels = classify(net, resizedTest);
true_test_labels = test.Labels;
accuracy_test = mean(true_test_labels == pred_test_labels);
confusionchart(confusionmat(true_test_labels, pred_test_labels))

sports-image-classification-matlab/
│
├─ report/
│   └─ Neural_computing.pdf
│
├─ .gitignore
├─ LICENSE
└─ README.md
