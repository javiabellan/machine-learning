# Data Visualization

## Common plot parameters

    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16

## Univariate plotting (1 column) with pandas

<table>
<tr>
<td><img src="https://i.imgur.com/skaZPhb.png" width="350px"/></td>
<td><img src="https://i.imgur.com/gaNttYd.png" width="350px"/></td>
<td><img src="https://i.imgur.com/pampioh.png"/></td>
<td><img src="https://i.imgur.com/OSbuszd.png"/></td>

<!--<td><img src="https://i.imgur.com/ydaMhT1.png" width="350px"/></td>
<td><img src="https://i.imgur.com/WLAqDSV.png" width="350px"/></td>
<td><img src="https://i.imgur.com/Tj2y9gH.png"/></td>
<td><img src="https://i.imgur.com/X0qXLCu.png"/></td>-->
</tr>
<tr>
<td style="font-weight:bold; font-size:16px;">Bar Chat</td>
<td style="font-weight:bold; font-size:16px;">Line Chart</td>
<td style="font-weight:bold; font-size:16px;">Area Chart</td>
<td style="font-weight:bold; font-size:16px;">Histogram</td>
</tr>
<tr>
<td>df.plot.bar()</td>
<td>df.plot.line()</td>
<td>df.plot.area()</td>
<td>df.plot.hist()</td>
</tr>
<tr>
<td>Good for nominal and small ordinal categorical data.</td>
<td colspan="2">Good for ordinal categorical and interval data.</td>
<td>Good for interval data.</td>
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

## Dimensionality reduction (Multivariate plotting in a 2D or 3D space)

- Principal Component Analysis (PCA)
- t-SNE


