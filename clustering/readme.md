# Clustering

Agrupar datos desetiquetados en grupos

* Hard Clustering: Un datapoint pertenece exclusivamente a un grupo.
* Soft Clustering: A un datpoint se le da la probabilidad a la que pertenece a cada grupo.

## Modelos

* Centroid models: O(N) pero hay que estableblecer K. 
  * **K-means**
  * **Mean-Shift**
  * Affitinty propagation
  * Gaussian Mixture model
* Connectivity models (hierarchical models): Cuando dentro de los grupos, hay subgrupos (ejmplo: animales). O(N^2)
  * **Agglomerative Hierarchical Clustering**
* Density Models: Buscan regiones densas
  * **DBSCAN**
  * OPTICS
* Distribution models: Probabilidad de un punto de pertenecer aun cluster: normal, gausiana...
  * **Expectation-Maximization (EM)**
* Otros
  * Dimensionality Reduction
    * tSNE
    * PCA
    * Kernel PCA
  * Laplacian Eigen Maps
  * Deep learning
    * Autoencoders
    * Self-organizing Maps
  * Probabilistic
    * LDA

## K means

Ventajas
* Simplicidad
* Eficicencia

Desventajas:
* Determinar manualmente k
* Senitivo a outliers
* Solo agrupa hyper-elipsoides (no agrupa formas raras)

## X means

## Mean-Shift

## Affitinty propagation

## DBScan

# Conclusión

![img](http://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png)



# Referencias

* [The 5 Clustering Algorithms Data Scientists Need to Know](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)
