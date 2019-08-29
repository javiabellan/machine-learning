<h1 align="center">Machine learning</h1>

### ML Pipeline
|   |                                                                    |                                            |
|---|--------------------------------------------------------------------|--------------------------------------------|
| 0 | 📊 [**Data visualization**](#-visualization-)                      | Plots for exploratory data analysis (EDA)  |
| 1 | 🛁 [**Data cleaning**](#-data-cleaning-)                           | Preprocess and clean the data.             |
| 2 | 🛠 [**Feature engineering**](#-feature-engineering-)                | Select and construct appropriate features. |
| 3 | 🔮 **Models**: [**Prediction**](#-prediction-models-), [**Clustering**](#-clustering-models-) | Select an appropriate model. |
| 4 | 🎯 [**Hyperparams optimization**](#-hyperparameters-optimization-) | Optimize model hyperparameters.            |
| 5 | 📏 **Metrics**: [**Classification**](#-Classification-metrics-), [**Regression**](#-Regression-metrics-) | Measure the model performance.  |
| 6 | ❓ [**Explainability**](#-explainability-)                         | Interpret the model.                       |
| all | 🍹 [**Auto Machine learning**](#-auto-machine-learning-)           | Automatic machine learning pipeline        |


### 📚 Python libraries

| Library                                                            | Description                                |    |
|--------------------------------------------------------------------|--------------------------------------------|----|
| 🔢 [**Numpy**](https://numpy.org)                                  | Vectors and matrices                       | ⭐ |
| 🐼 [**Pandas**](#data-manipulation-with-pandas)                    | Data manipulation                          | ⭐ |
| 📊 [**Matplotlib**](#visualization-with-matplotlib-and-seaborn)    | Data visualization                         | ⭐ |
| 📊 [**Seaborn**](#visualization-with-matplotlib-and-seaborn)       | Data visualization                         |    |
| 💡 [**Scikit learn**](https://scikit-learn.org)                     | Machine learning                           | ⭐ |
| 🔦 [**Pytorch**](https://pytorch.org)                              | Deep learning                              | ⭐ |
| 🌳 [**XGBoost**](https://github.com/dmlc/xgboost)                  | Gradient Boosting                          | ⭐ |
| 🌳 [**LightGBM**](https://github.com/Microsoft/LightGBM)           | Gradient Boosting                         |
| 🌳 [**CatBoost**](https://github.com/catboost/catboost)            | Gradient Boosting                         |
| 💧 [**H2O**](https://www.h2o.ai/products/h2o/)                      | Auto Machine learning                      | ⭐ |
| 🍵 [**TPOT**](https://github.com/EpistasisLab/tpot)              | Auto Machine learning                     |
| 💡 [**Auto Sklearn**](https://github.com/automl/auto-sklearn)       | Auto Machine learning                     |
| 📦 [**MLBox**](https://github.com/AxeldeRomblay/MLBox)             | Auto Machine learning                     |




  
----------------------------------------------------------------


# 🛁 Data cleaning [🔝](#machine-learning)

| Name           | Description                  | Options                          |
|:--------------:|------------------------------|----------------------------------|
| **Duplicates** | Repeated rows in the dataset | Remove                           |
| **Missings**   | No data on some features     | Remove row, remove feature, fill |
| **Ouliers**    | Rare or unexpected features  | Remove                           |

Check: https://github.com/tysonjens/Notes/edit/master/README.md

Split data into x, y for training and testing
```python
from sklearn.model_selection import train_test_split
## make a train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

Scaling - normalizes values of each feature, mean = 0, sd = 1
```python
## scale using pandas
df['var'] = ((df['var']-(df['var'].mean())))/(df['var'].std())
```

```Python
## Dummies in pandas
df = pd.DataFrame({'country': ['usa', 'canada', 'australia','japan','germany']})
pd.get_dummies(df,prefix=['country'])
```

```python
## Get Dummies, drop one
df = pd.get_dummies(df, drop_first=True)
```

```Python
# Binarizing
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
print(lb.transform((1,4)))
print(lb.classes_)
```

```python
# Ordinal Encoder transforms categorical features into int features
from sklearn.preprocessing import OrdinalEncoder
my_cat_feature = np.array(['Alpha', 'Boone', 'Kelli', 'Kelli', 'Boone', 'Tyson', 'Boone']).reshape(-1, 1)
encoder = OrdinalEncoder()
my_cat_feat_encoded = encoder.fit_transform(my_cat_feature)
my_cat_feat_encoded

## Outputs

array([[0.],
       [1.],
       [2.],
       [2.],
       [1.],
       [3.],
       [1.]])

```

```python
## One Hot Encoding takes a single categorical feature and converts it
## into several dummy columns
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
my_hot_encoded_dummy_cols = OneHotEncoder.fit_transform(my_cat_feature)
```

```python
## Imputation
## See number of nulls
test_scores.isnull().sum(0)

## Strategy could be 'mean', 'most_frequent'
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
X = [[np.nan, 1], [6, np.nan], [3, 6]]
imp.transform(X)

## outputs:
array([[ 4.        ,  1.        ],
       [ 6.        ,  3.66666667],
       [ 3.        ,  6.        ]])
```

```Python
def strfeat_to_intfeat(strfeat):
    ## input: list of features as strings
    ## output = list of same features now as integers as unique IDs
    valdict = {}
    intfeat = []
    for idx, i in enumerate(strfeat):
        if i in valdict:
            intfeat.append(valdict[i])
        else:
            valdict[i] = idx
            intfeat.append(idx)
    return intfeat
```

```Python
## Standard Scaler - you should always fit your scaler on training data,
## then apply it to the test data
scaler = StandardScaler().fit(X_train)
X_train_1 = scaler.transform(X_train)
X_test_1 = scaler.transform(X_test)
```

```python
## change string feature with multiple levels into numerical
def strfeat_to_intfeat(strfeat):
    ## input: list of features as strings
    ## output = list of same features now as integers as unique IDs
    valdict = {}
    intfeat = []
    for idx, i in enumerate(strfeat):
        if i in valdict:
            intfeat.append(valdict[i])
        else:
            valdict[i] = idx
            intfeat.append(idx)
    return intfeat
```

# 🛠 Feature engineering [🔝](#machine-learning)

| Name                       | Description              | Options                                                          |
|----------------------------|--------------------------|------------------------------------------------------------------|
| **Feature transformation** | Modidy existing features | Scaling, normalize, standarize, logarithim, ...                  |
| **Feature generation**     | Add useful features      | Modify to new, Combine features, Cluster some feature, ...       |
| **Feature selection**      | Remove useless features  | See feat importance, correlations, Dimensionality reduction, ... |

 TO DO: What is Latent feature discovery ??? 

- [Unbalanced](#): Deal with **imbalanced datasets**: Check [imbalanced-learn package](http://imbalanced-learn.org)
  - **Subsample majority class**. But you can lose important data.
  - **Oversample minority class**. But you can overfit.
  - **Weighted loss function** `CrossEntropyLoss(weight=[…])`


## Feature selection
Read [sklearn chapter](https://scikit-learn.org/stable/modules/feature_selection.html)

Reduce number of attributes.
- [**Feature selection**](https://scikit-learn.org/stable/modules/feature_selection.html)
- Wrapper: Su usa un classificador
  - MultiObjectiveEvolutionarySearch: Mejor para muchas generaciones. 10000 Evals
  - PSO: Particule Search optimization: Mejor para pocas generaciones.
  - RFE: Recursive feature elimination
  - SelectKBest
  - Variance Threshold
- Filters:
  - InfoGAIN: Cantidad de informacion
  - Correlation Featue Selection
  
#### Recursive Feature Elimination (RFE)   
At each iteration, select one feature to remove until there are n feature left*

```python
from sklearn.feature_selection import RFE
```

#### SelectKBest
The SelectKBest class just scores the features using a function and then removes all but the k highest scoring features.

```python
from sklearn.feature_selection import SelectKBest
```

#### Variance Threshold
Drop all features that dont meet a variance threshold

```python
from sklearn.feature_selection import VarianceThreshold
```

## Dimensionality reduction
> - https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
> - Read [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
> - [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)
> - [Matrix factorization](https://scikit-learn.org/stable/modules/decomposition.html)

Method       | Name                                          | Based in                   | Duration
:-----------:|-----------------------------------------------|----------------------------|-----
**PCA**      | Principal Component Analysis                  | Linear (maximize variance) | Fast
**t-SNE**    | t Stochastic Neighbor Embedding               | Neighbors |
**LargeVis** | LargeVis                                      | Neighbors |
**ISOMAP**   | t Stochastic Neighbor Embedding               | Neighbors |
**UMAP**     | Uniform Manifold Approximation and Projection | Neighbors |
**AE**       | Autoencoder (2 or 3 at hidden layer)          | Neural    |
**VAE**      | Variational Autoencoder                       | Neural    |
**LSA**      | Latent Semantic Analysis                      |           |
**SVD**      | Singular Value decomposition                  | Linear?   |
**LDA**      | Linear Discriminant Analysis                  | Linear    |
**MDS**      | Multidimensional Scaling                      |           |

Why reduce dimensions?
- Remove multicollinearity
- Deal with the curse of dimensionality
- Remove redundant features
- Interpretation & Visualization
- Make computations easier
- Identify Outliers


#### Principal Component Analysis (PCA)
a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. The first component is the most important one, followed by the second, then the third, and so on.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
```

#### T-SNE
Read [How to use t-SNE effectively](https://distill.pub/2016/misread-tsne)

```python
from sklearn.manifold import TSNE

tsne   = TSNE(random_state=0)
x_tsne = tsne.fit_transform(x)

# And plot it:
plt.scatter(x_tsne[:, 0], x_tsne[:, 1]);
```

#### Independent Component Analysis (ICA)
a statistical technique for revealing hidden factors that underlie sets of random variables, measurements, or signals.

#### Principal Component Regression (PCR)
a technique for analyzing multiple regression data that suffer from multicollinearity. The basic idea behind PCR is to calculate the principal components and then use some of these components as predictors in a linear regression model fitted using the typical least squares procedure.

#### Partial Least Squares Regression (PLSR)
PCR creates components to explain the observed variability in the predictor variables, without considering the response variable at all. On the other hand, PLSR does take the response variable into account, and therefore often leads to models that are able to fit the response variable with fewer components.

#### Sammon Mapping
an algorithm that maps a high-dimensional space to a space of lower dimensionality by trying to preserve the structure of inter-point distances in high-dimensional space in the lower-dimension projection. sometimes we have to ask the question “what non-linear transformation is optimal for some given dataset”. While PCA simply maximizes variance, sometimes we need to maximize some other measure that represents the degree to which complex structure is preserved by the transformation. Various such measures exist, and one of these defines the so-called Sammon Mapping. It is particularly suited for use in exploratory data analysis.

#### Multidimensional Scaling (MDS)
a means of visualizing the level of similarity of individual cases of a dataset.

#### Projection Pursuit
a type of statistical technique that involves finding the most “interesting” possible projections in multidimensional data. Often, projections which deviate more from a normal distribution are considered to be more interesting.

#### Linear Discriminant Analysis (LDA)
if you need a classification algorithm you should start with logistic regression. However, LR is traditionally limited to only two class classification problems. Now, if your problem involves more than two classes you should use LDA. LDA also works as a dimensionality reduction algorithm; it reduces the number of dimension from original to C — 1 number of features where C is the number of classes.

#### Mixture Discriminant Analysis (MDA) — It is an extension of linear discriminant analysis. Its a supervised method for classification that is based on mixture models.

#### Quadratic Discriminant Analysis (QDA)
Linear Discriminant Analysis can only learn linear boundaries, while Quadratic Discriminant Analysis is capable of learning quadratic boundaries (hence it is more flexible). Unlike LDA however, in QDA there is no assumption that the covariance of each of the classes is identical.

#### Flexible Discriminant Analysis (FDA)
a classification model based on a mixture of linear regression models, which uses optimal scoring to transform the response variable so that the data are in a better form for linear separation, and multiple adaptive regression splines to generate the discriminant surface.




# 🔮 Prediction models [🔝](#machine-learning)
> [TODO read](https://towardsdatascience.com/ml-algorithms-one-sd-%CF%83-74bcb28fafb6)

- Linear Models:         [scikit-learn chapeter](https://scikit-learn.org/stable/modules/linear_model.html)
- Decision Tree:         [scikit-learn chapeter]( https://scikit-learn.org/stable/modules/tree.html)
- Support Vector Machine: [scikit-learn chapeter]( https://scikit-learn.org/stable/modules/svm.html)
- Ensemble methods:       [scikit-learn chapeter](https://scikit-learn.org/stable/modules/ensemble.html)

### 🔮❓ Interpretable Models
Simple models. Good for starting point (baseline), understand the data, and create surrogate models of blackbox models.

| Interpretable Model | Linear | Monotone | Interaction | Task       |    |
|---------------------|--------|----------|-------------|------------|----|
| Linear Regression   | Yes    | Yes      | No          | regr       | ⭐ |
| Logistic Regression | No     | Yes      | No          | class      |    |
| Decision Tree       | No     | Some     | Yes         | class,regr | ⭐ |
| Decision Rules      |        |          |             |            |    |
| RuleFit             | Yes    | No       | Yes         | class,regr |    |
| Naive Bayes         | No     | Yes      | No          | class      |    |
| K-Nearest Neighbors | No     | No       | No          | class,regr |    |

### 🔮📦 Black Box Models
Better models

| Model                   |   |
|-------------------------|---|
| Support Vector Machine  |   |
| Random forest           | ⭐ |
| Extra trees             |   |
| Adaboost                |   |
| Gradient boosting (GBM) | ⭐⭐⭐ |
| Neural Network          | ⭐⭐   |


---

## Linear Models

#### Generalized Linear Models (GLMs)
Target does not follow a Gaussian distribution. Wrap the lineal reg. with a function.
- Binary category: **Logistic regression** (add sigmoid) and **Probit regression**
- Many categories: **Multinomial logistic regression** and **Multinomial probit regression**
- ordinal data: **Ordered probit regression**
- Discrete count: **Poisson regression**
- Time to the occurrence of an event:
- Very skewed outcome with a few very high values (household income).

The features interact:
- Adding interactions manually

Not linear:
- Feature tranformations (log, root, exp, ...)
- Feature categorization (new subfeatures)
- Generalized Additive Models (GAMs):
  Fit standard regression coefficients to dome variables and nonlinear spline functions to other variables.


#### Ordinary Least Squares Regression (OLSR)
A method in Linear Regression for estimating the unknown parameters by creating a model which will minimize the sum of the squared errors between the observed data and the predicted one (observed values and estimated values).

#### Linear Regression
used to estimate real values (cost of houses, number of calls, total sales etc.) based on continuous variable.

#### Logistic Regression
used to estimate discrete values ( Binary values like 0/1, yes/no, true/false) based on given set of independent variable

#### Stepwise Regression
adds features into your model one by one until it finds an optimal score for your feature set. Stepwise selection alternates between forward and backward, bringing in and removing variables that meet the criteria for entry or removal, until a stable set of variables is attained. Though, I haven’t seen too many articles about it and I heard couple of arguments that it doesn’t work.

#### Multivariate Adaptive Regression Splines (MARS)
a flexible regression method that searches for interactions and non-linear relationships that help maximize predictive accuracy. This algorithms is inherently nonlinear (meaning that you don’t need to adapt your model to nonlinear patterns in the data by manually adding model terms (squared terms, interaction effects)).

#### Locally Estimated Scatterplot Smoothing (LOESS)
a method for fitting a smooth curve between two variables, or fitting a smooth surface between an outcome and up to four predictor variables. The idea is that what if your data is not linearly distributed you can still apply the idea of regression. You can apply regression and it is called as locally weighted regression. You can apply LOESS when the relationship between independent and dependent variables is non-linear.

#### Quantile regression
???

## Regulated linear models (Penalized regression):
An extension made to linear models (typically regression methods) that penalizes models (penalty parameter) based on their complexity, favoring simpler models that are also better at generalizing.
- **L1** o **LASSO**: Least Absolute Shrinkage and Selection Operator. Good for feat selection
- **L2** o **RIDGE**: For robustness
- **Elastic Net**:
- **LARS**: Least-Angle Regression

#### Ridge Regression (L2 Regularization)
Its goal is to solve problems of data overfitting. A standard linear or polynomial regression model will fail in the case where there is high collinearity (the existence of near-linear relationships among the independent variables) among the feature variables. Ridge Regression adds a small squared bias factor to the variables. Such a squared bias factor pulls the feature variable coefficients away from this rigidness, introducing a small amount of bias into the model but greatly reducing the variance. The Ridge regression has one main disadvantage, it includes all n features in the final model.

#### LASSO Regression (L1 Regularization)
Least Absolute Shrinkage and Selection Operator (LASSO). In opposite to Ridge Regression, it only penalizes high coefficients. Lasso has the effect of forcing some coefficient estimates to be exactly zero when hyper parameter θ is sufficiently large. Therefore, one can say that Lasso performs variable selection producing models much easier to interpret than those produced by Ridge Regression.

```python
LogisticRegression(penalty='l1')
```

#### Elastic Net
Ccombines some characteristics from both lasso and ridge. Lasso will eliminate many features, while ridge will reduce the impact of features that are not important in predicting your y values. This algorithm reduces the impact of different features (like ridge) while not eliminating all of the features (like lasso).

#### Least-Angle Regression (LARS)
Similar to forward stepwise regression. At each step, it finds the predictor most correlated with the response. When multiple predictors having equal correlation exist, instead of continuing along the same predictor, it proceeds in a direction equiangular between the predictors.

## Decision Tree
- No need to normalize data.
- Algorithms:
  - **ID3**: Iterative Dichotomiser 3
  - **C4.5**: a newer version of ID3 (in weka is called J48)
  - **C5.0**
  - **CART**: Classification And Regression Trees.
  - **CHAID**: Chi-squared Automatic Interaction Detection
  - **Decision Stump**
  - **M5**

#### Iterative Dichotomiser 3 (ID3)
builds a tree top-down. It starts at the root and choose an attribute that will be tested at each node. Each attribute is evaluated through some statistical means in order to detect which attribute splits the dataset the best. The best attribute becomes the root, with its attribute values branching out. Then the process continues with the rest of the attributes. Once an attribute is selected, it is not possible to backtrack.

#### C4.5 and C5.0 (different versions of a powerful approach)
C4.5, Quinlan’s next iteration is a newer version of ID3. The new features (versus ID3) are: (i) accepts both continuous and discrete features; (ii) handles incomplete data points; (iii) solves over-fitting problem by bottom-up technique usually known as “pruning”; and (iv) different weights can be applied the features that comprise the training data. C5.0, the most recent Quinlan iteration. This implementation is covered by patent and probably as a result, is rarely implemented (outside of commercial software packages).

#### Classification and Regression Tree (CART)
CART is used as an acronym for the term decision tree. In general, implementing CART is very similar to implementing the above C4.5. The one difference though is that CART constructs trees based on a numerical splitting criterion recursively applied to the data, while the C4.5 includes the intermediate step of constructing rule sets.

### Chi-squared Automatic Interaction Detection (CHAID)
an algorithm used for discovering relationships between a categorical response variable and other categorical predictor variables. It creates all possible cross tabulations for each categorical predictor until the best outcome is achieved and no further splitting can be performed. CHAID builds a predictive model, or tree, to help determine how variables best merge to explain the outcome in the given dependent variable. In CHAID analysis, nominal, ordinal, and continuous data can be used, where continuous predictors are split into categories with approximately equal number of observations. It is useful when looking for patterns in datasets with lots of categorical variables and is a convenient way of summarizing the data as the relationships can be easily visualized.

#### Decision Stump
a ML model that is consisted of a one-level decision tree; a tree with one internal node (the root) which is connected to the terminal nodes (its leaves). This model makes a prediction based on the value of just a single input feature.

#### M5
M5 combines a conventional decision tree with the possibility of linear regression functions at the nodes. Besides accuracy, it can take tasks with very high dimension — up to hundreds of attributes. M5 model tree is a decision tree learner for regression task, meaning that it is used to predict values of numerical response variable Y. While M5 tree employs the same approach with CART tree in choosing mean squared error as impurity function, it does not assign a constant to the leaf node but instead it fit a multivariate linear regression model.


## Decision Rules
- **OneR**: Learns rules from a single feature. OneR is characterized by its simplicity, interpretability and its use as a benchmark.
- **Sequential covering**: General procedure that iteratively learns rules and removes the data points that are covered by the new rule.
- **Bayesian Rule Lists**: Combine pre-mined frequent patterns into a decision list using Bayesian statistics.
- RIPPER
- M5Rules
- PART
- JRip
- FURIA (fuzzy)

## Association rule learning
Given a set of transactions, the goal is to find rules that will predict the occurrences of an item based on the occurrences of other items in the transactions.
- Apriori algorithm
- Eclat algorithm
- FP (Frequent Pattern) Growth

#### Apriori
has great significance in data mining. It is useful in mining frequent itemsets (a collection of one or more items) and relevant association rules. You usually use this algorithm on a database that has a large number of transactions. For example, the items customers buy at a supermarket. The Apriori algorithm reduces the number of candidates with the following principle: If an itemset is frequent, ALL of its subsets are frequent.

#### Eclat (Equivalence Class Transformation)
the biggest difference from the Apriori algorithm is that it uses Depth First Search instead of Breadth First Search. In the Apriori algorithm, the element based on the product (shopping cart items 1, 2, 3, 3, etc.) is used, but in Eclat algorithm, the transaction is passed on by the elements (Shopping Cart 100,200 etc.).

#### FP (Frequent Pattern) Growth
helps perform a market basket analysis on transaction data. Basically, it’s trying to identify sets of products that are frequently bought together. FP-Growth is preferred to Apriori because Apriori takes more execution time for repeated scanning of the transaction dataset to mine the frequent items.


## Instance based models:
Instance and distances based. Utilidad: Conjunto multieditado y condensado: Para reducir el dataset y limparlo. Utilidad 2: Para pedecir atributos missing.

- **K Nearest Neighbors (KNN)**: Used in recommendation systems. k = 5, 10 or sqrt(Num samples).
- **Weighted KNN**: Closer samples are more imortant. Better than KNN.
- **Fuzzy KNN**: Sample pionts class labels are multiclass vetor (distance to class centroids).
- **Parzen**: Define a window size (with gaussian shape for ex.) and select those samples. (k would be variable).
- **Learning Vector Quantization (LVQ)**
- **Self-Organizing Map (SOM)**
- **Locally Weighted Learning (LWL)**


#### K-Nearest Neighbor (KNN)
can be used for both classiﬁcation and regression problems. KNN stores all available cases and classiﬁes new cases by a majority vote of its K neighbors. Predictions are made for a new data point by searching through the entire training set for the K most similar instances (the neighbors) and summarizing the output variable for those K instances. For regression problems, this might be the mean output variable, for classification problems this might be the mode (or most common) class value.

#### Learning Vector Quantization (LVQ)
A downside of K-Nearest Neighbors is that it hangs on to the entire training dataset. LVQ is an artificial neural network algorithm that allows you to choose how many training instances to hang onto and learns exactly what those instances should look like. If you discover that KNN gives good results on your dataset try using LVQ to reduce the memory requirements of storing the entire training dataset.

#### Self-Organizing Map (SOM)
an unsupervised deep learning model, mostly used for feature detection or dimensionality reduction. It outputs a 2D map for any number of indicators. SOM differ from other artificial neural networks as it apply competitive learning as opposed to error-correction learning (like backpropagation with gradient descent), and in the sense that they use a neighborhood function to preserve the topological properties of the input space.

#### Locally Weighted Learning (LWL)
The idea behind this algorithm is that instead of building a global model for the entire function space, for each point of interest we build a local model based on neighboring data of the query point. For this purpose, each data point becomes a weighting factor which expresses the influence of the data point for the prediction. Mainly, data points that are in the close neighborhood to the current query point are receiving a higher weight than data points which are far away.

## Bayesian models
- Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Averaged One-Dependence Estimators (AODE)
- Bayesian Belief Network (BBN)
- Bayesian Network (BN)

#### Naive Bayes
assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature (independence). Provides a way of calculating posterior probability P(c|x) from P(c), P(x) and P(x|c). Useful for very large data sets.

#### Gaussian Naive Bayes
assumes that the distribution of probability is Gaussian (normal). For continuous distributions, the Gaussian naive Bayes is the algorithm of choice.

#### Multinomial Naive Bayes
a specific instance of Naive Bayes where the P(Featurei|Class) follows multinomial distribution (word counts, probabilities, etc.). This is mostly used for document classification problem (whether a document belongs to the category of sports, politics, technology etc.). The features/predictors used by the classifier are the frequency of the words present in the document.

#### Averaged One-Dependence Estimators (AODE)
developed to address the attribute-independence problem of the naive Bayes classifier. AODE frequently develops considerably more accurate classifiers than naive Bayes with a small cost of a modest increase in the amount of computation.

#### Bayesian Belief Network (BBN)
a probabilistic graphical model (a type of statistical model) that represents a set of variables and their conditional dependencies via a directed acyclic graph (DAG). For example, a Bayesian network could represent the probabilistic relationships between diseases and symptoms. Given symptoms, the network can be used to compute the probabilities of the presence of various diseases. A BBN is a special type of diagram (called a directed graph) together with an associated set of probability tables.

#### Bayesian Network (BN)
the goal of Bayesian networks is to model conditional dependence, and therefore causation, by representing conditional dependence by edges in a directed graph. Using them, you can efficiently conduct inference on the random variables in the graph through the use of factors.

#### Hidden Markov models (HMM)
a class of probabilistic graphical model that give us the ability to predict a sequence of unknown (hidden) variables from a set of observed variables. For example, we can use it to predict the weather (hidden variable) based on the type of clothes that someone wears (observed). HMM can be viewed as a Bayes Net unrolled through time with observations made at a sequence of time steps being used to predict the best sequence of hidden states.

#### Conditional random fields (CRFs)
a classical machine learning model to train sequential models. It is a type of Discriminative classifier that model the decision boundary between the different classes. The difference between discriminative and generative models is that while discriminative models try to model conditional probability distribution, i.e., P(y|x), generative models try to model a joint probability distribution, i.e., P(x,y). Their underlying principle is that they apply Logistic Regression on sequential inputs. Hidden Markov Models share some similarities with CRFs, one in that they are also used for sequential inputs. CRFs are most used for NLP tasks.


## Support Vector Machines (SVM)
- with liear kernel
- with RBF kernel: Very good one


## Ensamble models
Stronger models.

- **Boosting**
- **Bootstrapped Aggregation (Bagging)**
- **Random Forest**: Rows & atribs bagging + Decision tress [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  - Deeper trees
- **Extra Trees**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- **Adaboost**
- **Stacked Generalization (blending)**
- **Gradient Boosting**: Works great with heterogeneous data and small datasets (unlike neural nets). [link1](http://explained.ai/gradient-boosting/index.html), [link2](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d), [link3](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
  - Gradient Boosting Machines (GBM)
  - Gradient Boosted Regression Trees (GBRT)
  - Tree depth from 3 to 6
  - [**XGBoost**](https://github.com/dmlc/xgboost), [**LightGBM**](https://github.com/Microsoft/LightGBM), [**CatBoost**](https://github.com/catboost/catboost) 💪 **Scikit-learn**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)


#### Random Forest
Random forests provide an improvement over bagged trees by way of a random small tweak that decorrelates the trees. As in bagging, we build a number forest of decision trees on bootstrapped training samples. But when building these decision trees, each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors. The split is allowed to use only one of those m predictors.

Ways to interpret feature impact:
- Partial Dependency Plot
- Permute a single feature
- Keep track of information gains due to each features
- Keep track of traffic that passes by each value.


#### Gradient boosting
- Works great with heterogeneous data and small datasets (unlike neural nets). [link1](http://explained.ai/gradient-boosting/index.html), [link2](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d), [link3](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
- Tree depth from 3 to 6
- [**XGBoost**](https://github.com/dmlc/xgboost), [**LightGBM**](https://github.com/Microsoft/LightGBM), [**CatBoost**](https://github.com/catboost/catboost) 💪 **Scikit-learn**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)



# 🔮 Clustering models [🔝](#machine-learning)
Separate data in groups, useful for labeling a dataset.
- Knowing K
  - **K-Means**
  - **k-Medians**
  - **Mean-Shift**
- Without knowing K
  - **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise. 

Methods to determine best k:
- Elbow Method
- Gap Method - like elbow method, but comparing with uniform
- Silhouette Score - (b-a) / max(a,b) where:
  - a is inter cluster distance,
  - b is next-nearest cluster centroid

#### K-Means
K means goal is to partition X data points into K clusters where each data point is assigned to its closest cluster. The idea is to minimize the sum of all squared distances within a cluster, for all clusters. A completely differerent algorithm than KNN (don’t confuse the two!).

#### single-linkage clustering
one of several methods of hierarchical clustering. It is based on grouping clusters in bottom-up fashion. In single-linkage clustering, the similarity of two clusters is the similarity of their most similar members.

#### K-Medians
a variation of K means algorithm. The idea is that instead of calculating the mean for each cluster (in order to determine its centroid), we calculate the median.

#### Expectation Maximization (EM)
it works similarly to K means except for the fact that the data is assigned to each cluster with the weights being soft probabilities instead of distances. It has the advantage that the model becomes generative as we define the probability distribution for each model.

#### Hierarchical Clustering
does not partition the dataset into clusters in a single step. Instead it involves multiple steps which run from a single cluster containing all the data points to N clusters containing single data point.

#### Fuzzy clustering
a form of clustering in which each data point can belong to more than one cluster.

#### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
used to separate clusters of high density from clusters of low density. DBSCAN requires just two parameters: the minimum distance between two points and the minimum number of points to form a dense region. Meaning, it groups together points that are close to each other (usually Euclidean distance) and a minimum number of points.

#### OPTICS (Ordering Points to Identify Cluster Structure)
the idea behind it is similar to DBSCAN, but it addresses one of DBSCAN’s major weaknesses: the problem of detecting meaningful clusters in data of varying density.

#### Non negative matrix factorization (NMF)
a Linear-algebraic model that factors high-dimensional vectors into a low-dimensionality representation. Similar to Principal component analysis (PCA), NMF takes advantage of the fact that the vectors are non-negative. By factoring them into the lower-dimensional form, NMF forces the coefficients to also be non-negative.

#### Latent Dirichlet allocation (LDA)
a type of probabilistic model and an algorithm used to discover the topics that are present in a corpus. For example, if observations are words collected into documents, to obtain the cluster assignments, it needs two probability values: P( word | topics), the probability of a word given topics. And P( topics | documents), the probability of topics given documents. These values are calculated based on an initial random assignment. Then, you iterate them for each word in each document, to decide their topic assignment.

####Gaussian Mixture Model (GMM)
Its goal is to find a mixture of multi-dimensional Gaussian probability distributions that best model any input dataset. It can be used for finding clusters in the same way that k-means does. The idea is quite simple, find the parameters of the Gaussians that best explain our data. We assume that the data is normal and we want to find parameters that maximize the likelihood of observing these data.


  
### Time series analysis
- Time series: Sequence of values of some feature (obtained in constant time periods).
- Goal: Get the forecast (predict future values).





# 🎯 Hyperparameters optimization [🔝](#machine-learning)

> TO-DO: Read A Comparative Study of Black-box Optimization Algorithms for Tuning of Hyper-parameters in Deep Neural Networks.

| Method    | Name                                                    | Type         | Stars   |
|------------|--------------------------------------------------------|--------------|---------|
| **GS**     | **Grid Search**                                        | Parallel     |         |
| **RS**     | **Random Search**                                      | Parallel     |         |
| **BO-GP**  | **Bayesian Optimization with Gaussian Processes**      | Sequential   | ⭐      |
| **PSO**    | **Particle Swarm optimization**                        | Evolutionary | ⭐      |
| **NM**     | **Nelder-Mead Algorithm**                              | ?            | ⭐⭐   |
| **TPE**    | **Tree of Parzen Estimators**                          | ?            | ⭐⭐⭐ |
|            | **Simulated Annealing**                                | ?            |         |
|            | **Gradient Descent**                                   | Sequential   |         |
| **CMA-ES** | **Covariance Matrix Adaptation Evolutionary Etrategy** | Evolutionary |         |

### Packages
- [Sklearn](https://scikit-learn.org):            GS, RS
- [Optunity](https://optunity.readthedocs.io):    GS, RS, NM, PSO and TPE
- GPyOpt:                                         BO-GP
- [BayesianOptimization](https://github.com/fmfn/BayesianOptimization): BO-GP
- [Hyperopt](https://hyperopt.github.io/hyperopt): RS, TPE

### Explanations
- Grid Search: Search over a discrete set of predefined hyperparameters values.
- Random Search: Provide a statistical distribution for each hyperparameter, for taking a random value.
- Gradient Descent: Optimize hyperparameters using gradient descent.
- Evolutionary: Uses evolutionary algorithms to search the space of possible hyperparameters.


# 📏 Classification Metrics [🔝](#machine-learning)

> - [Scikit-learn classification metrics](https://scikit-learn.org/stable/modules/classes.html#classification-metrics)
> - [H2O classification metric scores](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/scorers.html#classification)
> - [H2O classification metric plots](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/diagnosing.html#classification-metric-plots)

Check:
- Hinge loss (like in SVM)
- Square loss (like in ridge regression)
- Logistic loss or cross-entropy (like in logistic regression)
- Exponential loss (like in boosting)

| Score                         | Description                                  | Tip                          |
|-------------------------------|----------------------------------------------|------------------------------|
| [**Accuracy**](#accuracy)     | `# correctly predicted / # observations`     | Highly interpretable         |
| [**Precision**](#precision)   | `TP / TP + FP` = `TP / predicted possitives` |                              |
| [**Recall**](#recall)         | `TP / TP + FN` = `TP / actual possitives`    |                              |
| [**Fβ Score**](#fβ-Score)     | `(1+β²) * (Prec*Rec)/(β²*Prec+Rec)`          |                              |
| [**F05**](#f05-Score)         | `1.25 * (Prec*Rec)/(0.25*Prec+Rec)`          | Good when you want to give more weight to precision |
| [**F1**](#f1-Score)           | `2 * (Prec*Rec)/(Prec+Rec)`                  |                              |
| [**F2**](#f2-Score)           | `5 * (Prec*Rec)/(4*Prec+Rec)`                | Good when you want to give more weight to recall    |
| [**Dice**](#dice-Score)       |`2 * (Pred ∩ GT)/(Pred + GT)`                 |                              |
| [**Log loss**](#log-loss)     |                                              |                              |
| [**MCC**](#mcc)               | Matthews Correlation Coefficient             | Represents the confusion matrix. Good for imbalanced |
| [**AUC**](#auc)               | Area Under the roc Curve                     | Represent the ROC curve.      |
| [**AUCPR**](#aucpr)           | Area Under the precision-recall Curve        |                               |
| [**MACROAUC**](#macroauc)     | Macro average of Areas Under the roc Curves  | Good for imbalanced data      |

![img](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/_images/diagnosing.png)

| Classification Metric Plots     |   |
|---------------------------------|---|
| **Confusion Matrix**            | ⭐ |
| **ROC Curve**                   | ⭐ |
| **Precision-Recall Curve**      |   |
| **Cumulative Gains**            |   |
| **Lift Chart**                  |   |
| **Kolmogorov-Smirnov Chart**    |   |

### Example
Dataset with 5 disease images and 20 normal images. If the model predicts all images to be normal, its accuracy is 80%, and F1-score of such a model is 0.88


# 📏 Regression Metrics [🔝](#machine-learning)

| Scores    | Full name                         | Tip
|-----------|-----------------------------------|--------------------------------------------
| **ME**    | Mean Error  (or Mean Bias Error)  | It could determine if the model has positive bias or negative bias.
| **MAE**   | Mean Absolute Error               | The most simple.
| **MSE**   | Mean Squared Error                | Penalice large errors more than MAE.
| **MSLE**  | Mean Squared Log Error            |
| **MPE**   | Mean Percent Error                | Use when target values are across different scales
| **MAPE**  | Mean Absolute Percent Error       | Use when target values are across different scales
| **SMAPE** | Symmetric Mean Abs Percent Error  | Use when target values close to 0
| **MSPE**  | Mean Squared Percent Error        |
| **RMSE**  | Root Mean Squared Error ⭐        | Proportional to MSE.
| **RMSLE** | Root Mean Squared Log Error       | Not penalize large differences when both values are large numbers.
| **RMSPE** | Root Mean Squared Percent Error   | Use when target values are across different scales
| **R2**    | R² (coefficient of determination) |

⚠️ Note that **Squared** errors are sensitive to **outliers** (bad) because penalizes large errors by a lot more.

### Regression Metric Plots
- Actual vs Predicted
- Residual Plot with LOESS curve
- Residual Histogram

![img](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/_images/diagnosing_regression.png)


# ❓ Explainability [🔝](#machine-learning)
- [h2o blog](https://www.h2o.ai/blog/how-to-explain-a-model-with-h2o-driverless-ai)
- [h2o doc](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/interpreting.html)
- [**THE BOOK**](https://christophm.github.io/interpretable-ml-book)

| Technique                                                |
|----------------------------------------------------------|
| 1. Global Shapley Feature Importance                     |
| 2. Global Original Feature Importance                    |
| 3. Partial Dependence                                    |
| 4. Global Surrogate Decision Tree                        |
| 5. Global Interpretable Model                            |
| 6. Local Shapley Feature Importance                      |
| 7. Local Linear Explanations                             |
| 8. Local Surrogate Tree Decision Path                    |
| 9. Original Feature Individual Conditional Exception ICE |
| 10. Local Original Feature Importance                    |

![img](https://www.h2o.ai/wp-content/uploads/2019/02/1.-Global-Shapley-Feature-Importance.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/2.-Global-Original-Feature-Importance.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/3.-Partial-Dependence.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/4.-Global-Surrogate-Decision-Tree.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/5.-Global-Interpretable-Model.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/6.-Local-Shapley-Feature-Importance.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/7.-Local-Linear-Explanations.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/8.-Local-Surrogate-Tree-Decision-Path.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/9.-Original-Feature-Individual-Conditional-Exception-ICE.png)
![img](https://www.h2o.ai/wp-content/uploads/2019/02/10.-Local-Original-Feature-Importance.png)
---


# 🍹 Auto Machine learning [🔝](#machine-learning)
> - [**MLBox**](https://github.com/AxeldeRomblay/MLBox)
> - [**Auto Sklean**](https://github.com/automl/auto-sklearn)
> - [**TPOT**](https://github.com/EpistasisLab/tpot) ⭐
> - [**H20**](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) ⭐
> - Neural Architecture Search (NAS) for deep learning
>   - **DARTS**: Differentiable Architecture Search
>   - [**Uber Ludwig**](https://uber.github.io/ludwig/) ⭐
>   - [**Autokeras**](https://autokeras.com/)
> - References
>   - [Automl webpage](https://www.automl.org/automl)
>   - [Siraj video](https://youtu.be/jn-22XyKsgo)
>

### Neural Architecture Search (NAS)
- **DARTS**: Differentiable Architecture Search [*paper*](https://arxiv.org/abs/1806.09055), [DARTS in PyTorch](https://github.com/quark0/darts)
----------------------------------------------------------------


# 🌍 Real world applications [🔝](#machine-learning)
> - loss-given-default
> - probability of default
> - customer churn
> - campaign response
> - fraud detection
> - anti-money-laundering
> - predictive asset maintenance
> - References
>   - [link](https://www.knime.com/solutions)


# 🗞️ Data sources [🔝](#machine-learning)
- Files
  - CSV
  - Excel
  - Parquet (columnar storage file format of Hadoop)
  - Feather
  - Python datatable (.nff, .jay)
- No relational databases
  - MongoDB
  - Redis
- Relational Databases (SQL)
  - MySQL
- Big data
  - Hadoop (HDFS)
  - S3 (Amazon)
  - Azure Blob storage
  - Blue Data Tap
  - Google big query
  - Google cloud storage
  - kdb+
  - Minio
  - Snowflake
  
  
# 🐼 Data manipulation with [Pandas](https://pandas.pydata.org) [🔝](#machine-learning)
> - [**Kaggle learn Pandas**](https://www.kaggle.com/learn/pandas)

- Import pandas library `import pandas as pd`
- Read a CSV file into a pandas dataframe `df = pd.read_csv("file.csv")`
- Get dataframe info:
  - Show firt/last rows `df.head()` `df.tail()`
  - Get shape: `df.shape`. Get columns: `df.columns.tolist()`.
  - Print some info (like missings and types): `df.info()`
  - Has missings? `df.isnull().any().any()`
  - Describe numerical atributes `df.describe()`
  - Describe categorical atributes `df.describe(include=['object', 'bool'])`
- Do some data exploration
  - Get some column (return a series) `df["column"]`
  - Get some columns (return a df) `df[["column1", "column1"]]`
  - Apply function to column `.mean()` `.std()` `.median()` `.max()` `.min()` `.count()`
  - Count unique values `.value_counts()`
- Filter dataframe rows
  - One condition `df[df["column"]==1]`
  - Multiple conditions `df[(df["column1"]==1) & (df["column2"]=='No')]`
- Save it in a CSV [`df.to_csv("sub.csv", index=False)`](http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-store-in-csv)



# 📊 Visualization [🔝](#machine-learning)
Libraries: [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
> - [**Kaggle learn visualization**](https://www.kaggle.com/learn/data-visualization)
> - [**Python graph gallery**](https://python-graph-gallery.com)


#### [H2O available graphs](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/datasets.html#the-visualization-page):
- Correlated Scatterplot
- Spikey Histograms
- Skewed Histograms
- Varying Boxplots
- Heteroscedastic Boxplots
- Biplot (PCA points and arrows)
- Outliers
- Correlation Graph
- Parallel Coordinates Plot
- Radar Plot
- Data Heatmap
- Missing Values Heatmap
- Gaps Histogram

#### Types
- Univariate visualization
  - Histogram
  - Density plot
  - Box plot
  - Violin plot
- Bivariate visualization
- Multivariate visualization
  - Parallel coords
  - Radar chart


### Numerical data distribution
<table>
<tr>
    <td><a href="https://python-graph-gallery.com/histogram">
        <img src="https://python-graph-gallery.com/wp-content/uploads/HistogramBig-150x150.png" width="100px"/></td>
    <td><a href="https://python-graph-gallery.com/density-plot">
        <img src="https://python-graph-gallery.com/wp-content/uploads/DensityBig-150x150.png"   width="100px"/></td>
    <td><a href="https://python-graph-gallery.com/boxplot">
        <img src="https://python-graph-gallery.com/wp-content/uploads/Box1Big-150x150.png"      width="100px"/></td>
    <td><a href="https://python-graph-gallery.com/violin-plot">
        <img src="https://python-graph-gallery.com/wp-content/uploads/ViolinBig-150x150.png"    width="100px"/></td>
</tr>
<tr>
    <td>Histogram</td>
    <td>Density plot</td>
    <td>Box plot</td>
    <td>Violin plot</td>
</tr>
<tr>
    <td>df.plot.hist()<br>sns.distplot()</td>
    <td>df.plot.kde()<br>sns.kdeplot()</td>
    <td>df.plot.box()<br>sns.boxplot()</td>
    <td>sns.violinplot()</td>
</tr>
</table>

### Comparing numerical features
<table>
<tr>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/ScatterPlotBig-150x150.png"      width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/ScatterConnectedBig-150x150.png" width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/BubblePlotBig-150x150.png"       width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/HeatmapBig-150x150.png"          width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/2dDensityBig-150x150.png"        width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/CorrelogramBig-150x150.png"      width="100px"/></td>
</tr>
<tr>
<td>Scatter plot</td>
<td>Line plot</td>
<td>Bubble plot</td>
<td>Heatmap</td>
<td>Density plot 2D</td>
<td>Correlogram</td>
</tr>
<tr>
<td>df.plot.scatter()<br>plt.scatter()<br>sns.scatterplot()</td>
<td></td>
<td></td>
<td>plt.imshow(np)<br>sns.heatmap(df)
</td>
<td>df.plot.hexbin()</td>
<td>scatter_matrix(df)<br>sns.pairplot()</td>
</tr>
</table>


### Ranking
<table>
<tr>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/BarBig-150x150.png"      width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/LollipopBig-150x150.png" width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/Parallel1Big-150x150.png"       width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/SpiderBig-150x150.png"          width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/WordcloudBig-150x150.png"        width="100px"/></td>
</tr>
<tr>
<td>Bar plot</td>
<td>Lollipop plot</td>
<td>Parallel coords.</td>
<td>Radar chart</td>
<td>Word cloud</td>
</tr>
<tr>
<td>plt.scatter()<br>sns.scatterplot()</td>
<td></td>
<td>parallel_coordinates(df, 'cls')</td>
<td></td>
<td></td>
</tr>
</table>

### Part of a whole

<table>
<tr>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/StackedBig-150x150.png"      width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/PieBig-150x150.png" width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/DoughnutBig-150x150.png"       width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/DendrogramBig-150x150.png"          width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/TreeBig-150x150.png"        width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/VennBig-150x150.png"      width="100px"/></td>
</tr>
<tr>
<td>Stacked bar plot</td>
<td>Pie chart</td>
<td>Donut chart</td>
<td>Dendrogram</td>
<td>Treemap</td>
<td>Venn diagram</td>
</tr>
</table>


### Evolution

<table>
<tr>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/LineBig-150x150.png"      width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/AreaBig-150x150.png" width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/StackedAreaBig-150x150.png"       width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/StreamBig-150x150.png"          width="100px"/></td>
</tr>
<tr>
<td>Line chart</td>
<td>Area chart</td>
<td>Stacked area chart</td>
<td>Stream graph</td>
</tr>
</table>

---

# Others TODO:
- [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html)
- Prueba U de Mann-Whitney
- Prueba t de Student
- Metrica Kappa
- Self Organizing Map
- Restricted boltzmann machine: Como el autoencoder pero va y vuelve
- Competitive learning
- Hebbian learning
- Evolutionary algorithms
  - Check [Platypus](https://platypus.readthedocs.io/en/latest/index.html)


# Resources
- [ML overview](https://vas3k.com/blog/machine_learning/)

<img align="right" width="400" src="https://www.kaggle.com/static/images/education/homepage-illustration.png">
