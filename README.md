# DBSCAN of Multi-Slice Clustering for three-order Tensor



This code implements the extension of the multi-slice clustering algorithm [1]. Our algorithm, called MSC-DBSCAN, aims to extract the different clusters of slices that lie in the different subspaces from the data when the data set is a sum of $r$ rank-one tensors $(r > 0)$. Our algorithm uses the  same input hyperparameters as the MSC algorithm.


Let $\mathcal{T} \in \mathbb{R}^{m_1\times m_2\times m_3}$  the tensor dataset, defined by $\mathcal{T} = \mathcal{X} + \mathcal{Z}$ where $\mathcal{X}$ is the signal tensor and $\mathcal{Z}$ is the noise tensor. We assume that

$$
\mathcal{T} = \mathcal{X} + \mathcal{Z} = \sum_{i = 1}^{r}\gamma_i\, \mathbf{w}_{i}\otimes \mathbf{u}_{i}\otimes\mathbf{v}_{i} + \mathcal{Z}
$$

With $r=2$, the similarity matrix of the slices in one mode seems as in the picture below:

![an image](msc-extension.png)


In this case: 
* The MSC returns the two clusters as a one single cluster.
* The MSC extension is able to separate the two disjoint clusters.


We evaluate the comparaison of the MSC and MSC-DBSCAN with the following code

```python
import rmse_msc_and_mscDbscan as rmse
rmse_MSC, rmse_Dbscan = [], []
for _ in range(20):
    a = rmse.run()
    rmse_MSC.append(a[0])
    rmse_Dbscan.append(a[1])
rmse.rmse_boxplot(rmse_MSC, rmse_Dbscan)
```

We also evaluate the quality of the clustering result according to the value of the signal strength

```python
import Ari_mscDbscan
a = Ari_mscDbscan.run()
Ari_mscDbscan.ari_plot(a[0], a[1])
```

The last experiment show the similarity of the output of  MSC and MSC-DBSCAN with the flow injection analysis (FIA) dataset [2].

```python
import experiment_fia_data
experiment_fia_data.run()
```



[1] D. F. Andriantsiory, J. B. Geloun, and M. Lebbah, “Multi-slice clustering for 3-order tensor,” in 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 2021, pp. 173–178.

[2] R. C. Norgaard L, “Rank annihilation factor analysis applied to flow injection analysis with photodiode-array detection,” 1994, chemometrics and Intelligent Laboratory, Systems 23:107.