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
> - [Missings imputation](#)
> - [Outliers removal](#)


# 🛠 Feature engineering [🔝](#machine-learning)
> - [Scaling (normalize, standarize, logs, ...)](#)
> - [Feature selection](#feature-selection)
> - [Dimensionality reduction](#dimensionality-reduction)
> - [Unbalanced](#)

- Deal with **imbalanced datasets**: Check [imbalanced-learn package](http://imbalanced-learn.org)
  - **Subsample majority class**. But you can lose important data.
  - **Oversample minority class**. But you can overfit.
  - **Weighted loss function** `CrossEntropyLoss(weight=[…])`


### Feature selection
Reduce number of attributes.
- [**Feature selection**](https://scikit-learn.org/stable/modules/feature_selection.html)
- Wrapper: Su usa un classificador
  - MultiObjectiveEvolutionarySearch: Mejor para muchas generaciones. 10000 Evals
  - PSO: Particule Search optimization: Mejor para pocas generaciones.
  - RFE: Recursive feature elimination
- Filters:
  - InfoGAIN: Cantidad de informacion
  - Correlation Featue Selection

### Dimensionality reduction
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

### PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
```


### T-SNE

Read [How to use t-SNE effectively](https://distill.pub/2016/misread-tsne)

```python
from sklearn.manifold import TSNE

tsne   = TSNE(random_state=0)
x_tsne = tsne.fit_transform(x)

# And plot it:
plt.scatter(x_tsne[:, 0], x_tsne[:, 1]);
```





# 🔮 Prediction models [🔝](#machine-learning)
> - [Predictive models (classification and regresion)](#)
>   - [Linear](#)
>   - [Decision tree](#)
>   - [Random forest](#)
>   - [Gradient boosting](#)
>   - [Support vector machine](#)
> - [Clustering models](#clustering)
> - [Time series models](#)

### Simple models
Good for starting point (baseline), meta-features (input to other models), stacking (final output).
- **Logistic regression**: For classification
- **Linear regression**: For regrssion
- Instance and distances based:
  - **K nearest neighbors (KNN)**: Used in recommendation systems. k = 5, 10 or sqrt(Num samples).
  - **Weighted KNN**: Closer samples are more imortant. Better than KNN.
  - **Fuzzy KNN**: Sample pionts class labels are multiclass vetor (distance to class centroids).
  - **Parzen**: Define a window size (with gaussian shape for ex.) and select those samples. (k would be variable).
  - Utilidad: Conjunto multieditado y condensado: Para reducir el dataset y limparlo.
  - Utilidad 2 : Para pedecir atributos missing
- **Decision tree**: J48, C4.5 No need to normalize data.
- **Support Vector Machines (SVM)**
  - with liear kernel
  - with RBF kernel: Very good one
- **Naive bayes**
- **Rule based**: PART, JRip, FURIA (fuzzy)


### Ensamble models
Stronger models.
- **Random forest**: Rows & atribs bagging + Decision tress [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  - Deeper trees
- **Extra trees**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- **Adaboost**
- **Gradient boosting**: Works great with heterogeneous data and small datasets (unlike neural nets). [link1](http://explained.ai/gradient-boosting/index.html), [link2](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d), [link3](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
  - Tree depth from 3 to 6
  - [**XGBoost**](https://github.com/dmlc/xgboost), [**LightGBM**](https://github.com/Microsoft/LightGBM), [**CatBoost**](https://github.com/catboost/catboost) 💪 **Scikit-learn**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

### Gradient boosting
- Works great with heterogeneous data and small datasets (unlike neural nets). [link1](http://explained.ai/gradient-boosting/index.html), [link2](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d), [link3](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
- Tree depth from 3 to 6
- [**XGBoost**](https://github.com/dmlc/xgboost), [**LightGBM**](https://github.com/Microsoft/LightGBM), [**CatBoost**](https://github.com/catboost/catboost) 💪 **Scikit-learn**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)



# 🔮 Clustering models [🔝](#machine-learning)
Separate data in groups, useful for labeling a dataset.
- Knowing K
  - **K-Means**
  - **Mean-Shift**
- Without knowing K
  - **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise. 
  
  
### Time series analysis
- Time series: Sequence of values of some feature (obtained in constant time periods).
- Goal: Get the forecast (predict future values).





# 🎯 Hyperparameters optimization [🔝](#machine-learning)
> - Grid search
> - Random search
> - Bayesian optimization ⭐
> - Particle swarm optimization (PSO)
> - Simulated Annealing
> - Gradient descent
> - Evolutionary algorithm

Method             | Description                                                                            | Library 
:------------------|----------------------------------------------------------------------------------------|--------
**Grid Search**    | Search over a discrete set of predefined hyperparameters values.                       | [Sklearn](https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)
**Random Search**  | Provide a statistical distribution for each hyperparameter, for taking a random value. |  [Sklearn](https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization)
**Bayesian**       | Use past evaluation results to choose the next values to evaluate. **`THE BEST`**      | [link](https://github.com/fmfn/BayesianOptimization)
**Gradient-Based** | Optimize hyperparameters using gradient descent.                                       | ?
**Evolutionary**   | Uses evolutionary algorithms to search the space of possible hyperparameters.          | ?

Note that **grid** and **random** search can be **paralelized**, others methods can not.



# 📏 Classification Metrics [🔝](#machine-learning)

> - [Scikit-learn classification metrics](https://scikit-learn.org/stable/modules/classes.html#classification-metrics)
> - [H2O classification metric scores](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/scorers.html#classification)
> - [H2O classification metric plots](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/diagnosing.html#classification-metric-plots)

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


# ❓ Explainability [🔝](#machine-learning)
- [link](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/interpreting.html)
- [**THE BOOK**](https://christophm.github.io/interpretable-ml-book)



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
