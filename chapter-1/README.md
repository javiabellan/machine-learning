# Machine learning

## Chapter 1: Introduction

### What is machine learning?

![ML](https://www.marketsimplified.com/wp-content/uploads/2017/04/ml_vs_ai.jpg)

First is necesary to understand the diffencence between ML and AI:

 * **Artificial Intelligence** is any technique which enables computers to mimic human behabiors.
 * **Machine Learning** is a technique that allows computers to automatically learn by exploring some data.
 
The difference resides that AI is a very old concept (50s) that allow computers to look intelligent by implementing an 	explicit set of rules. And ML appears in the 80s by implementing algorithms that automatically learn by defining only some parameters (the input data and the desired goal). So ML is a subset of AI. According to wikipedia:

> Machine learning is an application of artificial intelligence that automates analytical **model** building by using algorithms that iteratively **learn from data** without being explicitly programmed where to look.

So basically we have **data** and we want to build a **model** that learns by itself.

Pretty cool right? But wait, what is data? and how we build model?

![i have no idea what i'm doing](https://cdn-images-1.medium.com/max/455/1*snTXFElFuQLSFDnvZKJ6IA.png)

---

### What is data?

Data is everything you can imagine: images, music, stock prices, etc. Usually we need a lot of data to build a good model. This is very simililar as how humans learn: the more data a human sees, the more intelligent it will become.

![think](http://i0.kym-cdn.com/photos/images/facebook/001/217/711/afd.jpg_large)

### What is a model?

That is the intelligence part, following with the human analogy, the model would be the brain. It is the part that process the data and learns from it.

There are a lot of machine learning models. We are going to see only the most populars ones.

![mind map](https://jixta.files.wordpress.com/2015/11/machinelearningalgorithms.png)


# Predictive models

These models are funcions of type **f(x) = y**, where **x** is the input and **y** is the prediction of the output. There are 2 types: regression and classification.

## Regression models

 * **Target y is continuous**
 * Popular methods
   * Linear regression
   * Generalize additive model

## Classification models

 * **Target y is categorical**
 * Popular methods
   * Logistic regression
   * Suppert vector machines (SVM)
   * Decision tree
   * Random forest
