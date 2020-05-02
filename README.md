# Detection of Onsets in String Instrument Music Using Convolutional Neural Networks

Final Year Project
## Supervisor
Dr.Peter Jancovic
## Student
Yuxuan 
at 
University of Birmingham
July 2019- July 2020

## Abstract

The main approach of this Project is to apply CNN modelling to onset detection, especially analyse the performance of learning on string instrument music detection. The project will try to generate an explicit comparable python programming for onset detection, and applied to string instrument music. we evaluated our models on datasets with different combinations: model trained with different tolerance, model trained with different dataset, and different models trained on same dataset. Our results show that different CNN Architectures has different advantages and disadvantages. The mixed dataset had better evaluation result. The models trained with fuzzier labels often
have considerable improvement on string music detection. We found that the model trained by adding asymmetric tolerance after onset label had the best performance.

![](./figure/frontend.png)
## Front to End Analysis

## Train
python main.py --train 0:8 --epochs 100
## Evaluate
python main.py --evaluate 0:1
