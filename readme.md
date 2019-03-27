<h1 align="center">Machine learning</h1>


<img align="right" width="500" src="https://www.kaggle.com/static/images/education/homepage-illustration.png">

> #### Exploratory Data Analysis
> 1. [**Data manipulation with Pandas**](#1-know-the-basics)
> 2. [**Visualization with Matplotlib & Seaborn**](#Visualization)
>
> #### Preprocessing
> 4. [**Feature selection**](#3-improve-generalization-and-avoid-overfitting-try-in-that-order)
> 5. [**Clustering**](#clustering)
>
> #### Models
> 3. [**Decision trees**](#2-choose-training-hyperparams)
> 3. [**Random forest**](#2-choose-training-hyperparams)
> 3. [**Gradient boosting**](#2-choose-training-hyperparams)
> 3. [**Support vector machine**](#2-choose-training-hyperparams)
> 3. [**Time series**](#2-choose-training-hyperparams)
>
> #### Validation and metrics

# 1. Data manipulation with [Pandas](https://pandas.pydata.org)
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

# 2. Visualization with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
> - [**Kaggle learn visualization**](https://www.kaggle.com/learn/data-visualization)
> - [**Python graph gallery**](https://python-graph-gallery.com)

### Numerical data distribution
<table>
<tr>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/HistogramBig-150x150.png" width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/DensityBig-150x150.png"   width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/Box1Big-150x150.png"      width="100px"/></td>
<td><img src="https://python-graph-gallery.com/wp-content/uploads/ViolinBig-150x150.png"    width="100px"/></td>
</tr>
<tr>
<td style="font-weight:bold; font-size:16px;">Histogram</td>
<td style="font-weight:bold; font-size:16px;">Density plot</td>
<td style="font-weight:bold; font-size:16px;">Box plot</td>
<td style="font-weight:bold; font-size:16px;">Violin plot</td>
</tr>
<tr>
<td>df.plot.hist()</td>
<td>df.plot.kde()</td>
<td>df.plot.box()<br>sns.boxplot(df)</td>
<td>sns.violinplot(df)</td>
</tr>
</table>

### Correlation

- **PCA**:
- **T-SNE**:


# Preprocessing
- **Feature extraction**
  - Color features
  - Texture features
- [**Feature selection**](https://scikit-learn.org/stable/modules/feature_selection.html): Reduce number of attributes.
  - Wrapper: Su usa un classificador
    - MultiObjectiveEvolutionarySearch: Mejor para muchas generaciones. 10000 Evals
    - PSO: Particule Search optimization: Mejor para pocas generaciones.
    - RFE: Recursive feature elimination
  - Filters:
    - InfoGAIN: Cantidad de informacion
    - Correlation Featue Selection


# Clustering
Separate data in groups, useful for labeling a dataset.
- Knowing K
  - **K-Means**
  - **Mean-Shift**
- Without knowing K
  - **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise. 




# Simple models
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


# Ensamble models
Stronger models.
- **Random forest**: Rows & atribs bagging + Decision tress [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  - Deeper trees
- **Extra trees**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- **Adaboost**
- **Gradient boosting**: Works great with heterogeneous data and small datasets (unlike neural nets). [link1](http://explained.ai/gradient-boosting/index.html), [link2](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d), [link3](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
  - Tree depth from 3 to 6
  - [**XGBoost**](https://github.com/dmlc/xgboost), [**LightGBM**](https://github.com/Microsoft/LightGBM), [**CatBoost**](https://github.com/catboost/catboost) 💪 **Scikit-learn**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

# Gradient boosting
- Works great with heterogeneous data and small datasets (unlike neural nets). [link1](http://explained.ai/gradient-boosting/index.html), [link2](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d), [link3](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
- Tree depth from 3 to 6
- [**XGBoost**](https://github.com/dmlc/xgboost), [**LightGBM**](https://github.com/Microsoft/LightGBM), [**CatBoost**](https://github.com/catboost/catboost) 💪 **Scikit-learn**: [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)


# 9. Time series analysis
- Time series: Sequence of values of some feature (obtained in constant time periods).
- Goal: Get the forecast (predict future values).

# Others
- Self Organizing Map
- [Autoencoder](/teoría/modelos/autoencoder.md): Para comprimir información
- Restricted boltzmann machine: Como el autoencoder pero va y vuelve
- competitive learning
- Hebbian learning
