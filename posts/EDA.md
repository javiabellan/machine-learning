# Data Visualization


## Import packages

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Common plot parameters

    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16

## Univariate plotting: Categorical data

<table>
<tr>
<td><img src="https://pandas.pydata.org/pandas-docs/stable/_images/series_pie_plot.png" width="300px"/></td>
<td><img src="https://i.imgur.com/skaZPhb.png" width="350px"/></td>
<td><img src="https://i.imgur.com/gaNttYd.png" width="350px"/></td>
<td><img src="https://i.imgur.com/pampioh.png"/></td>
</tr>
<tr>
<td style="font-weight:bold; font-size:16px;">Pie chart</td>
<td style="font-weight:bold; font-size:16px;">Bar chart</td>
<td style="font-weight:bold; font-size:16px;">Line chart</td>
<td style="font-weight:bold; font-size:16px;">Area chart</td>
</tr>
<tr>
<td>df.plot.pie()</td>
<td>df.plot.bar()</td>
<td>df.plot.line()</td>
<td>df.plot.area()</td>
</tr>
<tr>
<td colspan="2">Good for categorical data.</td>
<td colspan="2">Good for ordinal categorical data.</td>
</tr>
</table>


## Univariate plotting: Numerical data

<table>
<tr>
<td><img src="https://i.imgur.com/OSbuszd.png" width="450px"/></td>
<td><img src="https://pandas.pydata.org/pandas-docs/stable/_images/kde_plot.png" /></td>
<td><img src="https://s3.amazonaws.com/stackabuse/media/seaborn-library-data-visualization-python-part-1-13.png" /></td>
<td><img src="https://i.stack.imgur.com/DhyZK.png" /></td>
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
<td>df.plot.box()
sns.boxplot(df)</td>
<td>sns.violinplot(df)</td>
</tr>
</table>


## Bivariate plotting (2 columns) with pandas

<table>
<tr>
<td><img src="https://i.imgur.com/bBj1G1v.png" width="350px"/></td>
<td><img src="https://i.imgur.com/ChK9zR3.png" width="350px"/></td>
<td><img src="https://i.imgur.com/KBloVHe.png" width="350px"/></td>
<td><img src="https://i.imgur.com/C7kEWq7.png" width="350px"/></td>
</tr>
<tr>
<td style="font-weight:bold; font-size:16px;">Scatter Plot</td>
<td style="font-weight:bold; font-size:16px;">Hex Plot</td>
<td style="font-weight:bold; font-size:16px;">Stacked Bar Chart</td>
<td style="font-weight:bold; font-size:16px;">Bivariate Line Chart</td>
</tr>
<tr>
<td>df.plot.scatter()</td>
<td>df.plot.hexbin()</td>
<td>df.plot.bar(stacked=True)</td>
<td>df.plot.line()</td>
</tr>
<tr>
<td>Good for interval and some nominal categorical data.</td>
<td>Good for interval and some nominal categorical data.</td>
<td>Good for nominal and ordinal categorical data.</td>
<td>Good for ordinal categorical and interval data.</td>
</tr>
</table>




## Multivariate plotting (N columns)

<table>
<tr>
<td><img src="https://i.imgur.com/gJ65O47.png" width="350px"/></td>
<td><img src="https://i.imgur.com/3qEqPoD.png" width="350px"/></td>
<td><img src="https://i.imgur.com/1fmV4M2.png" width="350px"/></td>
<td><img src="https://i.imgur.com/H20s88a.png" width="350px"/></td>
</tr>
<tr>
<td style="font-weight:bold; font-size:16px;">Multivariate Scatter Plot</td>
<td style="font-weight:bold; font-size:16px;">Grouped Box Plot</td>
<td style="font-weight:bold; font-size:16px;">Correlation matrix</td>
<td style="font-weight:bold; font-size:16px;">Parallel Coordinates</td>
</tr>
<tr>
<td>df.plot.scatter()</td>
<td>df.plot.box()</td>
<td>cm = df[numerical].corr()
sns.heatmap(cm)</td>
<td>pd.plotting.parallel_coordinates</td>
</tr>
<!--
<tr>
<td>Good for interval and some nominal categorical data.</td>
<td>Good for interval and some nominal categorical data.</td>
<td>Good for nominal and ordinal categorical data.</td>
<td>Good for ordinal categorical and interval data.</td>
</tr>
-->
</table>

- Facets

# PCA

Principal Component Analysis

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
```

# t-SNE

Its basic idea is simple: find a projection for a high-dimensional feature space onto a 2D (or 3D) space, such that:
- Those points that were far apart in the initial n-dimensional space will end up far apart on the plane.
- Those that were originally close would remain close to each other.

Essentially, neighbor embedding is a search for a new and less-dimensional data representation that preserves neighborship of examples. It takes some take to compute the representation.

Data need to be normalized.

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```
Now, let's build a t-SNE representation:

```python
from sklearn.manifold import TSNE

tsne = TSNE(random_state=0)
tsne_repr = tsne.fit_transform(X_scaled)

# and plot it:
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], alpha=.5);
```


Let’s color this t-SNE representation according to the class.
```python
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=df['Churn'].map({False: 'blue', True: 'orange'}), alpha=.5);
```


