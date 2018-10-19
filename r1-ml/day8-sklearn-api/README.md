# scikit-learn Python 中的机器学习
---

简单高效的数据挖掘和数据分析工具

可供大家使用，可在各种环境中重复使用

建立在 NumPy，SciPy 和 matplotlib 上

开放源码，可商业使用 - BSD license


## 分类

识别某个对象属于哪个类别
* 应用: 垃圾邮件检测，图像识别
* 算法: SVM, nearest neighbors, random forest, …

## 回归

预测与对象相关联的连续值属性
* 应用: 药物反应，股价
* 算法: SVR, ridge regression, Lasso, …

## 聚类

将相似对象自动分组
* 应用: 客户细分，分组实验结果
* 算法: k-Means, spectral clustering, mean-shift, …

## 降维

减少要考虑的随机变量的数量
* 应用: 可视化，提高效率
* 算法: PCA, feature selection, non-negative matrix factorization.

## 模型选择

比较，验证，选择参数和模型
* 目标: 通过参数调整提高精度
* 模型: grid search, cross validation, metrics.
预处理

## 特征提取和归一化

* 应用: 把输入数据（如文本）转换为机器学习算法可用的数据
* 算法: preprocessing, feature extraction.

# 源码安装 scikit-learn

```sh
$ python3 -m venv sklearn0.20.0 # 创建 sklearn 的虚拟环境
$ source sklearn0.20.0/bin/activate # 激活虚拟环境
(sklearn0.20.0) $ wget -c https://files.pythonhosted.org/packages/0f/d7/136a447295adade38e7184618816e94190ded028318062a092daeb972073/scikit-learn-0.20.0.tar.gz # https://pypi.org/project/scikit-learn/#files
(sklearn0.20.0) $ tar xzvf scikit-learn-0.20.0.tar.gz # 解压
(sklearn0.20.0) $ cd scikit-learn-0.20.0/
(sklearn0.20.0) $ pip install numpy scipy cython pytest matplotlib # 安装依赖
(sklearn0.20.0) $ pip install --editable . # 安装 sklearn
(sklearn0.20.0) $ pytest sklearn # 测试
```

# API参考[](http://scikit-learn.org/stable/modules/classes.html#api-reference "永久链接到这个标题")

这是scikit-learn的类和函数参考。有关详细信息，请参阅[完整的用户指南](http://scikit-learn.org/stable/user_guide.html#user-guide)，因为类和功能原始规格可能不足以提供有关其用途的完整指南。有关API中重复的概念的参考，请参阅[通用术语和API元素术语表](http://scikit-learn.org/stable/glossary.html#glossary)。

## [`sklearn.base`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.base "sklearn.base")：基类和实用函数[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.base "永久链接到这个标题")

所有估计器的基类。

### [基类](http://scikit-learn.org/stable/modules/classes.html#base-classes "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`base.BaseEstimator`](http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator "sklearn.base.BaseEstimator") | scikit-learn中所有估计器的基类 |
| [`base.BiclusterMixin`](http://scikit-learn.org/stable/modules/generated/sklearn.base.BiclusterMixin.html#sklearn.base.BiclusterMixin "sklearn.base.BiclusterMixin") | Mixin类适用于scikit-learn中的所有双向估计器 |
| [`base.ClassifierMixin`](http://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin "sklearn.base.ClassifierMixin") | Mixin类适用于scikit-learn中的所有分类器。 |
| [`base.ClusterMixin`](http://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html#sklearn.base.ClusterMixin "sklearn.base.ClusterMixin") | Mixin类用于scikit-learn中的所有聚类估计器。 |
| [`base.DensityMixin`](http://scikit-learn.org/stable/modules/generated/sklearn.base.DensityMixin.html#sklearn.base.DensityMixin "sklearn.base.DensityMixin") | Mixin类适用于scikit-learn中的所有密度估计器。 |
| [`base.RegressorMixin`](http://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html#sklearn.base.RegressorMixin "sklearn.base.RegressorMixin") | Mixin类用于scikit-learn中的所有回归估计器。 |
| [`base.TransformerMixin`](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin "sklearn.base.TransformerMixin") | Mixin课程适用于scikit-learn中的所有变换器。 |

### [函数](http://scikit-learn.org/stable/modules/classes.html#functions "永久链接到这个标题")


| 类型 | 说明 |
|:--|:--|
| [`base.clone`](http://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html#sklearn.base.clone "sklearn.base.clone")(estimator[, safe]) | 构造具有相同参数的新估计器。 |
| [`base.is_classifier`](http://scikit-learn.org/stable/modules/generated/sklearn.base.is_classifier.html#sklearn.base.is_classifier "sklearn.base.is_classifier")(estimator) | 如果给定的估计器（可能）是分类器，则返回True。 |
| [`base.is_regressor`](http://scikit-learn.org/stable/modules/generated/sklearn.base.is_regressor.html#sklearn.base.is_regressor "sklearn.base.is_regressor")(estimator) | 如果给定的估计器（可能）是回归器，则返回True。 |
| [`config_context`](http://scikit-learn.org/stable/modules/generated/sklearn.config_context.html#sklearn.config_context "sklearn.config_context")（** NEW_CONFIG） | 用于全局scikit-learn配置的上下文管理器 |
| [`get_config`](http://scikit-learn.org/stable/modules/generated/sklearn.get_config.html#sklearn.get_config "sklearn.get_config")（） | 检索配置的当前值 [`set_config`](http://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config "sklearn.set_config") |
| [`set_config`](http://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config "sklearn.set_config")（[assume_finite，working_memory]） | 设置全局scikit-learn配置 |
| [`show_versions`](http://scikit-learn.org/stable/modules/generated/sklearn.show_versions.html#sklearn.show_versions "sklearn.show_versions")（） | 打印有用的调试信息 |

## [`sklearn.calibration`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration "sklearn.calibration")：概率校准[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration "永久链接到这个标题")

预测概率的校准。

**用户指南：**有关详细信息，请参阅[概率校准](http://scikit-learn.org/stable/modules/calibration.html#calibration)部分。


| 类型 | 说明 |
|:--|:--|
| [`calibration.CalibratedClassifierCV`](http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV "sklearn.calibration.CalibratedClassifierCV")（[...]） | 使用isotonic regression 或 sigmoid进行概率校准。 |


| 类型 | 说明 |
|:--|:--|
| [`calibration.calibration_curve`](http://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve "sklearn.calibration.calibration_curve")（y_true，y_prob） | 计算校准曲线的真实和预测概率。 |

## [`sklearn.cluster`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster "sklearn.cluster")：聚类[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster "永久链接到这个标题")

该[`sklearn.cluster`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster "sklearn.cluster")模块收集了流行的无监督聚类算法。

**用户指南：**有关详细信息，请参阅 [“聚类”](http://scikit-learn.org/stable/modules/clustering.html#clustering)部分。

### 类[](http://scikit-learn.org/stable/modules/classes.html#classes "永久链接到这个标题")


| 类型 | 说明 |
|:--|:--|
| [`cluster.AffinityPropagation`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation "sklearn.cluster.AffinityPropagation")([damping, …]) | AP聚类。 |
| [`cluster.AgglomerativeClustering`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering "sklearn.cluster.AgglomerativeClustering")（[...]） | 凝聚聚类 |
| [`cluster.Birch`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch "sklearn.cluster.Birch")（[threshold，branching_factor，...]） | 实现Birch聚类算法。 |
| [`cluster.DBSCAN`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN "sklearn.cluster.DBSCAN")（[eps，min_samples，metric，...]） | 从矢量数组或距离矩阵执行DBSCAN聚类。 |
| [`cluster.FeatureAgglomeration`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration "sklearn.cluster.FeatureAgglomeration")（[n_clusters，...]） | 特征聚集 |
| [`cluster.KMeans`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans "sklearn.cluster.KMeans")（[n_clusters，init，n_init，...]） | K-Means聚类 |
| [`cluster.MiniBatchKMeans`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans "sklearn.cluster.MiniBatchKMeans")（[n_clusters，init，...]） | Mini-Batch K-Means聚类 |
| [`cluster.MeanShift`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift "sklearn.cluster.MeanShift")([bandwidth, seeds, …]) | 使用flat核的均值偏移聚类。 |
| [`cluster.SpectralClustering`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering "sklearn.cluster.SpectralClustering")（[n_clusters，...]） | 谱聚类 将聚类应用于规范化拉普拉斯(laplacian)的谱。 |

### 函数[](http://scikit-learn.org/stable/modules/classes.html#id1 "永久链接到这个标题")


| 类型 | 说明 |
|:--|:--|
| [`cluster.affinity_propagation`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.affinity_propagation.html#sklearn.cluster.affinity_propagation "sklearn.cluster.affinity_propagation")（S [，...]） | 对数据执行AP聚类。 |
| [`cluster.dbscan`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html#sklearn.cluster.dbscan "sklearn.cluster.dbscan")（X [，eps，min_samples，...]） | 从矢量数组或距离矩阵执行DBSCAN聚类。 |
| [`cluster.estimate_bandwidth`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html#sklearn.cluster.estimate_bandwidth "sklearn.cluster.estimate_bandwidth")(X[, quantile, …])	 | 估计要使用均值平移算法的带宽。 |
| [`cluster.k_means`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html#sklearn.cluster.k_means "sklearn.cluster.k_means")（X，n_clusters [，...]） | K均值聚类算法。 |
| [`cluster.mean_shift`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.mean_shift.html#sklearn.cluster.mean_shift "sklearn.cluster.mean_shift")(X[, bandwidth, seeds, …])	 | 使用平面内核执行数据的均值漂移聚类。 |
| [`cluster.spectral_clustering`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.spectral_clustering.html#sklearn.cluster.spectral_clustering "sklearn.cluster.spectral_clustering")(affinity[, …]) | 将聚类应用于规范化拉普拉斯的投影。 |
| [`cluster.ward_tree`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.ward_tree.html#sklearn.cluster.ward_tree "sklearn.cluster.ward_tree")(X[, connectivity, …]) | 基于特征矩阵的Ward聚类。 |

## [`sklearn.cluster.bicluster`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster.bicluster "sklearn.cluster.bicluster")：双向聚类[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster.bicluster "永久链接到这个标题")

双向聚类算法。

作者：Kemal Eren许可证：BSD 3条款

**用户指南：**有关详细信息，请参阅 [“Biclustering”](http://scikit-learn.org/stable/modules/biclustering.html#biclustering)部分。

### 类[](http://scikit-learn.org/stable/modules/classes.html#id2 "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`SpectralBiclustering`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.bicluster.SpectralBiclustering.html#sklearn.cluster.bicluster.SpectralBiclustering "sklearn.cluster.bicluster.SpectralBiclustering")（[n_clusters，method，...]） | 光谱双聚（Kluger，2003）。 |
| [`SpectralCoclustering`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.bicluster.SpectralCoclustering.html#sklearn.cluster.bicluster.SpectralCoclustering "sklearn.cluster.bicluster.SpectralCoclustering")（[n_clusters，...]） | 频谱协同聚类算法（Dhillon，2001）。 |

## [`sklearn.compose`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose "sklearn.compose")：复合估计器[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose "永久链接到这个标题")

用于构建具有变换器的复合模型的元估计器

除了当前的内容，该模块最终将成为Pipeline和FeatureUnion的翻新版本的所在地。

**用户指南：**有关详细信息，请参阅[管道和复合估计器](http://scikit-learn.org/stable/modules/compose.html#combining-estimators)部分。

| 类型 | 说明 |
|:--|:--|
| [`compose.ColumnTransformer`](http://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer "sklearn.compose.ColumnTransformer")(transformers[, …]) | 将变换器应用于数组或pandas DataFrame的列。 |
| [`compose.TransformedTargetRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html#sklearn.compose.TransformedTargetRegressor "sklearn.compose.TransformedTargetRegressor")（[...]） | 元估计在回归目标上回归。 |

| 类型 | 说明 |
|:--|:--|
| [`compose.make_column_transformer`](http://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer "sklearn.compose.make_column_transformer")（......） | 从给定的变换器构造ColumnTransformer。 |

## [`sklearn.covariance`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance "sklearn.covariance")：协方差估计器[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance "永久链接到这个标题")

该[`sklearn.covariance`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance "sklearn.covariance")模块包括用于在给定一组点的情况下稳健地估计特征的协方差的方法和算法。还估计定义为协方差的倒数的精度矩阵。协方差估计与高斯图形模型理论密切相关。

**用户指南：**有关详细信息，请参阅[协方差估计](http://scikit-learn.org/stable/modules/covariance.html#covariance)部分。

| 类型 | 说明 |
|:--|:--|
| [`covariance.EmpiricalCovariance`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance "sklearn.covariance.EmpiricalCovariance")（[...]） | 最大似然协方差估计 |
| [`covariance.EllipticEnvelope`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope "sklearn.covariance.EllipticEnvelope")（[...]） | 用于检测高斯分布式数据集中的异常值的对象。 |
| [`covariance.GraphicalLasso`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html#sklearn.covariance.GraphicalLasso "sklearn.covariance.GraphicalLasso")（[alpha，mode，...]） | 利用l1惩罚估计量的稀疏逆协方差估计。 |
| [`covariance.GraphicalLassoCV`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLassoCV.html#sklearn.covariance.GraphicalLassoCV "sklearn.covariance.GraphicalLassoCV")（[alphas，...]） | 稀疏逆协方差w /交叉验证的l1惩罚选择 |
| [`covariance.LedoitWolf`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html#sklearn.covariance.LedoitWolf "sklearn.covariance.LedoitWolf")（[store_precision，...]） | LedoitWolf Estimator |
| [`covariance.MinCovDet`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html#sklearn.covariance.MinCovDet "sklearn.covariance.MinCovDet")（[store_precision，...]） | 最小协方差行列式（MCD）：协方差的稳健估计。 |
| [`covariance.OAS`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html#sklearn.covariance.OAS "sklearn.covariance.OAS")（[store_precision，...]） | Oracle近似收缩估计器 |
| [`covariance.ShrunkCovariance`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html#sklearn.covariance.ShrunkCovariance "sklearn.covariance.ShrunkCovariance")（[...]） | 具有收缩的协方差估计 |

| 类型 | 说明 |
|:--|:--|
| [`covariance.empirical_covariance`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html#sklearn.covariance.empirical_covariance "sklearn.covariance.empirical_covariance")（X[， …]） | 计算最大似然协方差估计 |
| [`covariance.graphical_lasso`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.graphical_lasso.html#sklearn.covariance.graphical_lasso "sklearn.covariance.graphical_lasso")（emp_cov，alpha [，...]） | l1-惩罚协方差估计 |
| [`covariance.ledoit_wolf`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html#sklearn.covariance.ledoit_wolf "sklearn.covariance.ledoit_wolf")（X [，assume_centered，...]） | 估计缩小的Ledoit-Wolf协方差矩阵。 |
| [`covariance.oas`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.oas.html#sklearn.covariance.oas "sklearn.covariance.oas")（X [，assume_centered]） | 估计与Oracle近似收缩算法的协方差。 |
| [`covariance.shrunk_covariance`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.shrunk_covariance.html#sklearn.covariance.shrunk_covariance "sklearn.covariance.shrunk_covariance")（emp_cov [，...]） | 计算在对角线上收缩的协方差矩阵 |

## [`sklearn.cross_decomposition`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition "sklearn.cross_decomposition")：交叉分解[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition "永久链接到这个标题")

**用户指南：**有关详细信息，请参阅[交叉分解](http://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition)部分。

| 类型 | 说明 |
|:--|:--|
| [`cross_decomposition.CCA`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html#sklearn.cross_decomposition.CCA "sklearn.cross_decomposition.CCA")（[n_components，...]） | CCA典型相关分析。 |
| [`cross_decomposition.PLSCanonical`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSCanonical.html#sklearn.cross_decomposition.PLSCanonical "sklearn.cross_decomposition.PLSCanonical")（[...]） | PLSCanonical实现了原始Wold算法[Tenenhaus 1998] p.204的2个块规范PLS，在[Wegelin 2000]中称为PLS-C2A。 |
| [`cross_decomposition.PLSRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html#sklearn.cross_decomposition.PLSRegression "sklearn.cross_decomposition.PLSRegression")（[...]） | PLS回归 |
| [`cross_decomposition.PLSSVD`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html#sklearn.cross_decomposition.PLSSVD "sklearn.cross_decomposition.PLSSVD")（[n_components，...]） | 偏最小二乘SVD |

## [`sklearn.datasets`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets "sklearn.datasets")：数据集[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets "永久链接到这个标题")

该[`sklearn.datasets`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets "sklearn.datasets")模块包括用于加载数据集的实用程序，包括加载和获取常用参考数据集的方法。它还具有一些人工数据生成器。

**用户指南：**有关详细信息，请参阅 [“数据集装入实用程序”](http://scikit-learn.org/stable/datasets/index.html#datasets)部分。

### 装载机[](http://scikit-learn.org/stable/modules/classes.html#loaders "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`datasets.clear_data_home`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.clear_data_home.html#sklearn.datasets.clear_data_home "sklearn.datasets.clear_data_home")（[data_home]） | 删除数据主缓存的所有内容。 |
| [`datasets.dump_svmlight_file`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.dump_svmlight_file.html#sklearn.datasets.dump_svmlight_file "sklearn.datasets.dump_svmlight_file")（X，y，f [，...]） | 以svmlight / libsvm文件格式转储数据集。 |
| [`datasets.fetch_20newsgroups`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups "sklearn.datasets.fetch_20newsgroups")（[data_home，...]） | 从20个新闻组数据集（分类）加载文件名和数据。 |
| [`datasets.fetch_20newsgroups_vectorized`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html#sklearn.datasets.fetch_20newsgroups_vectorized "sklearn.datasets.fetch_20newsgroups_vectorized")（[...]） | 加载20个新闻组数据集并将其矢量化为令牌计数（分类）。 |
| [`datasets.fetch_california_housing`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing "sklearn.datasets.fetch_california_housing")（[...]） | 加载加州住房数据集（回归）。 |
| [`datasets.fetch_covtype`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html#sklearn.datasets.fetch_covtype "sklearn.datasets.fetch_covtype")（[data_home，...]） | 加载隐藏数据集（分类）。 |
| [`datasets.fetch_kddcup99`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html#sklearn.datasets.fetch_kddcup99 "sklearn.datasets.fetch_kddcup99")（[subset，data_home，...]） | 加载kddcup99数据集（分类）。 |
| [`datasets.fetch_lfw_pairs`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html#sklearn.datasets.fetch_lfw_pairs "sklearn.datasets.fetch_lfw_pairs")（[子集，...]） | 加载标签面在野外（LFW）对数据集（分类）。 |
| [`datasets.fetch_lfw_people`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html#sklearn.datasets.fetch_lfw_people "sklearn.datasets.fetch_lfw_people")（[data_home，...]） | 加载标签面在野外（LFW）人数据集（分类）。 |
| [`datasets.fetch_olivetti_faces`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html#sklearn.datasets.fetch_olivetti_faces "sklearn.datasets.fetch_olivetti_faces")（[data_home，...]） | 从AT＆T（分类）加载Olivetti面数据集。 |
| [`datasets.fetch_openml`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml "sklearn.datasets.fetch_openml")([name, version, …]) | 按名称或数据集ID从openml获取数据集。 |
| [`datasets.fetch_rcv1`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html#sklearn.datasets.fetch_rcv1 "sklearn.datasets.fetch_rcv1")（[data_home，subset，...]） | 加载RCV1多标记数据集（分类）。 |
| [`datasets.fetch_species_distributions`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_species_distributions.html#sklearn.datasets.fetch_species_distributions "sklearn.datasets.fetch_species_distributions")（[...]） | 菲利普斯等物种分布数据集的载体。 |
| [`datasets.get_data_home`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.get_data_home.html#sklearn.datasets.get_data_home "sklearn.datasets.get_data_home")（[data_home]） | 返回scikit-learn数据目录的路径。 |
| [`datasets.load_boston`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston "sklearn.datasets.load_boston")（[return_X_y]） | 加载并返回波士顿房价数据集（回归）。 |
| [`datasets.load_breast_cancer`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer "sklearn.datasets.load_breast_cancer")（[return_X_y]） | 加载并返回乳腺癌威斯康星数据集（分类）。 |
| [`datasets.load_diabetes`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes "sklearn.datasets.load_diabetes")（[return_X_y]） | 加载并返回糖尿病数据集（回归）。 |
| [`datasets.load_digits`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits "sklearn.datasets.load_digits")（[n_class，return_X_y]） | 加载并返回数字数据集（分类）。 |
| [`datasets.load_files`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html#sklearn.datasets.load_files "sklearn.datasets.load_files")（container_path [，...]） | 加载带有类别的文本文件作为子文件夹名称。 |
| [`datasets.load_iris`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris "sklearn.datasets.load_iris")（[return_X_y]） | 加载并返回虹膜数据集（分类）。 |
| [`datasets.load_linnerud`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud "sklearn.datasets.load_linnerud")（[return_X_y]） | 加载并返回linnerud数据集（多元回归）。 |
| [`datasets.load_sample_image`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html#sklearn.datasets.load_sample_image "sklearn.datasets.load_sample_image")（IMAGE_NAME） | 加载单个样本图像的numpy数组 |
| [`datasets.load_sample_images`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html#sklearn.datasets.load_sample_images "sklearn.datasets.load_sample_images")（） | 加载样本图像以进行图像处理。 |
| [`datasets.load_svmlight_file`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html#sklearn.datasets.load_svmlight_file "sklearn.datasets.load_svmlight_file")（f [，n_features，...]） | 将svmlight / libsvm格式的数据集加载到稀疏CSR矩阵中 |
| [`datasets.load_svmlight_files`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_files.html#sklearn.datasets.load_svmlight_files "sklearn.datasets.load_svmlight_files")（文件[，...]） | 以SVMlight格式从多个文件加载数据集 |
| [`datasets.load_wine`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine "sklearn.datasets.load_wine")（[return_X_y]） | 加载并返回葡萄酒数据集（分类）。 |
| [`datasets.mldata_filename`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.mldata_filename.html#sklearn.datasets.mldata_filename "sklearn.datasets.mldata_filename")（数据名） | DEPRECATED：mldata_filename在0.20版本中已弃用，将在版本0.22中删除 |

### 样本生成器[](http://scikit-learn.org/stable/modules/classes.html#samples-generator "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`datasets.make_biclusters`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_biclusters.html#sklearn.datasets.make_biclusters "sklearn.datasets.make_biclusters")（形状，n_clusters） | 生成具有恒定块对角线结构的阵列，用于双向聚类。 |
| [`datasets.make_blobs`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs "sklearn.datasets.make_blobs")（[n_samples，n_features，...]） | 生成各向同性高斯blob用于聚类。 |
| [`datasets.make_checkerboard`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_checkerboard.html#sklearn.datasets.make_checkerboard "sklearn.datasets.make_checkerboard")（形状，n_clusters） | 生成具有用于双向聚类的块棋盘结构的数组。 |
| [`datasets.make_circles`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles "sklearn.datasets.make_circles")（[n_samples，shuffle，...]） | 在2d中制作一个包含较小圆圈的大圆圈。 |
| [`datasets.make_classification`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification "sklearn.datasets.make_classification")（[n_samples，...]） | 生成随机的n级分类问题。 |
| [`datasets.make_friedman1`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html#sklearn.datasets.make_friedman1 "sklearn.datasets.make_friedman1")（[n_samples，...]） | 生成“弗里德曼＃1”回归问题 |
| [`datasets.make_friedman2`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman2.html#sklearn.datasets.make_friedman2 "sklearn.datasets.make_friedman2")（[n_samples，noise，...]） | 生成“弗里德曼＃2”回归问题 |
| [`datasets.make_friedman3`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman3.html#sklearn.datasets.make_friedman3 "sklearn.datasets.make_friedman3")（[n_samples，noise，...]） | 生成“弗里德曼＃3”回归问题 |
| [`datasets.make_gaussian_quantiles`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles "sklearn.datasets.make_gaussian_quantiles")（[意思， …]） | 通过分位数生成各向同性高斯和标签样本 |
| [`datasets.make_hastie_10_2`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html#sklearn.datasets.make_hastie_10_2 "sklearn.datasets.make_hastie_10_2")（[n_samples，...]） | 生成Hastie等人使用的二进制分类数据。 |
| [`datasets.make_low_rank_matrix`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_low_rank_matrix.html#sklearn.datasets.make_low_rank_matrix "sklearn.datasets.make_low_rank_matrix")（[n_samples，...]） | 生成具有钟形奇异值的大多数低秩矩阵 |
| [`datasets.make_moons`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons "sklearn.datasets.make_moons")（[n_samples，shuffle，...]） | 制作两个交错的半圈 |
| [`datasets.make_multilabel_classification`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html#sklearn.datasets.make_multilabel_classification "sklearn.datasets.make_multilabel_classification")（[...]） | 生成随机多标签分类问题。 |
| [`datasets.make_regression`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression "sklearn.datasets.make_regression")（[n_samples，...]） | 生成随机回归问题。 |
| [`datasets.make_s_curve`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html#sklearn.datasets.make_s_curve "sklearn.datasets.make_s_curve")（[n_samples，noise，...]） | 生成S曲线数据集。 |
| [`datasets.make_sparse_coded_signal`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_coded_signal.html#sklearn.datasets.make_sparse_coded_signal "sklearn.datasets.make_sparse_coded_signal")（n_samples，...） | 生成信号作为字典元素的稀疏组合。 |
| [`datasets.make_sparse_spd_matrix`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_spd_matrix.html#sklearn.datasets.make_sparse_spd_matrix "sklearn.datasets.make_sparse_spd_matrix")（[dim，...]） | 生成稀疏对称确定正矩阵。 |
| [`datasets.make_sparse_uncorrelated`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html#sklearn.datasets.make_sparse_uncorrelated "sklearn.datasets.make_sparse_uncorrelated")（[...]） | 使用稀疏不相关设计生成随机回归问题 |
| [`datasets.make_spd_matrix`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_spd_matrix.html#sklearn.datasets.make_spd_matrix "sklearn.datasets.make_spd_matrix")（n_dim [，random_state]） | 生成随机对称正定矩阵。 |
| [`datasets.make_swiss_roll`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html#sklearn.datasets.make_swiss_roll "sklearn.datasets.make_swiss_roll")（[n_samples，noise，...]） | 生成瑞士卷数据集。 |

## [`sklearn.decomposition`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition "sklearn.decomposition")：矩阵分解[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition "永久链接到这个标题")

该[`sklearn.decomposition`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition "sklearn.decomposition")模块包括矩阵分解算法，包括PCA，NMF或ICA。该模块的大多数算法可以被视为降维技术。

**用户指南：**有关详细信息，请参阅[组件中](http://scikit-learn.org/stable/modules/decomposition.html#decompositions)的[分解信号（矩阵分解问题）](http://scikit-learn.org/stable/modules/decomposition.html#decompositions)部分。

| 类型 | 说明 |
|:--|:--|
| [`decomposition.DictionaryLearning`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning "sklearn.decomposition.DictionaryLearning")（[...]） | 字典学习 |
| [`decomposition.FactorAnalysis`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis "sklearn.decomposition.FactorAnalysis")（[n_components，...]） | 因子分析（FA） |
| [`decomposition.FastICA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA "sklearn.decomposition.FastICA")（[n_components，...]） | FastICA：独立分量分析的快速算法。 |
| [`decomposition.IncrementalPCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA "sklearn.decomposition.IncrementalPCA")（[n_components，...]） | 增量主成分分析（IPCA）。 |
| [`decomposition.KernelPCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA "sklearn.decomposition.KernelPCA")（[n_components，...]） | 核主成分分析（KPCA） |
| [`decomposition.LatentDirichletAllocation`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation "sklearn.decomposition.LatentDirichletAllocation")（[...]） | 具有在线变分贝叶斯算法的潜在Dirichlet分配 |
| [`decomposition.MiniBatchDictionaryLearning`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html#sklearn.decomposition.MiniBatchDictionaryLearning "sklearn.decomposition.MiniBatchDictionaryLearning")（[...]） | 小批量字典学习 |
| [`decomposition.MiniBatchSparsePCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html#sklearn.decomposition.MiniBatchSparsePCA "sklearn.decomposition.MiniBatchSparsePCA")（[...]） | 小批量稀疏主成分分析 |
| [`decomposition.NMF`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF "sklearn.decomposition.NMF")（[n_components，init，...]） | 非负矩阵分解（NMF） |
| [`decomposition.PCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA "sklearn.decomposition.PCA")（[n_components，copy，...]） | 主成分分析（PCA） |
| [`decomposition.SparsePCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA "sklearn.decomposition.SparsePCA")（[n_components，...]） | 稀疏主成分分析（SparsePCA） |
| [`decomposition.SparseCoder`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparseCoder.html#sklearn.decomposition.SparseCoder "sklearn.decomposition.SparseCoder")（dictionary[，...]） | 稀疏编码 |
| [`decomposition.TruncatedSVD`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD "sklearn.decomposition.TruncatedSVD")（[n_components，...]） | 使用截断的SVD（aka LSA）降低尺寸。 |

| 类型 | 说明 |
|:--|:--|
| [`decomposition.dict_learning`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.dict_learning.html#sklearn.decomposition.dict_learning "sklearn.decomposition.dict_learning")（X，n_components，...） | 解决字典学习矩阵分解问题。 |
| [`decomposition.dict_learning_online`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.dict_learning_online.html#sklearn.decomposition.dict_learning_online "sklearn.decomposition.dict_learning_online")（X[， …]） | 在线解决字典学习矩阵分解问题。 |
| [`decomposition.fastica`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.fastica.html#sklearn.decomposition.fastica "sklearn.decomposition.fastica")（X [，n_components，...]） | 执行快速独立分量分析。 |
| [`decomposition.sparse_encode`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.sparse_encode.html#sklearn.decomposition.sparse_encode "sklearn.decomposition.sparse_encode")（X，字典[，...]） | 稀疏编码 |

## [`sklearn.discriminant_analysis`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis "sklearn.discriminant_analysis")：判别分析[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis "永久链接到这个标题")

线性判别分析与二次判别分析

**用户指南：**有关详细信息，请参阅 [“线性和二次判别分析”](http://scikit-learn.org/stable/modules/lda_qda.html#lda-qda)部分。

| 类型 | 说明 |
|:--|:--|
| [`discriminant_analysis.LinearDiscriminantAnalysis`](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis "sklearn.discriminant_analysis.LinearDiscriminantAnalysis")（[...]） | 线性判别分析 |
| [`discriminant_analysis.QuadraticDiscriminantAnalysis`](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis")（[...]） | 二次判别分析 |

## [`sklearn.dummy`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy "sklearn.dummy")：虚拟估计器[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy "永久链接到这个标题")

**用户指南：**有关详细信息，请参阅[模型评估：量化预测质量](http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation)部分。

| 类型 | 说明 |
|:--|:--|
| [`dummy.DummyClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier "sklearn.dummy.DummyClassifier")（[strategy，......]） | DummyClassifier是一个使用简单规则进行预测的分类器。 |
| [`dummy.DummyRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor "sklearn.dummy.DummyRegressor")([strategy, constant, …]) | DummyRegressor是一个使用简单规则进行预测的回归量。 |

| 类型 | 说明 |
|:--|:--|

## [`sklearn.ensemble`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble "sklearn.ensemble")：合奏方法[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble "永久链接到这个标题")

该[`sklearn.ensemble`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble "sklearn.ensemble")模块包括用于分类，回归和异常检测的基于集合的方法。

**用户指南：**有关详细信息，请参阅 [“集合方法”](http://scikit-learn.org/stable/modules/ensemble.html#ensemble)部分。

| 类型 | 说明 |
|:--|:--|
| [`ensemble.AdaBoostClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier "sklearn.ensemble.AdaBoostClassifier")（[...]） | AdaBoost分类器。 |
| [`ensemble.AdaBoostRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor "sklearn.ensemble.AdaBoostRegressor")（[base_estimator，...]） | 一个AdaBoost回归量。 |
| [`ensemble.BaggingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier "sklearn.ensemble.BaggingClassifier")（[base_estimator，...]） | Bagging分类器。 |
| [`ensemble.BaggingRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor "sklearn.ensemble.BaggingRegressor")（[base_estimator，...]） | Bagging回归量。 |
| [`ensemble.ExtraTreesClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier "sklearn.ensemble.ExtraTreesClassifier")（[...]） | 一个额外的树分类器。 |
| [`ensemble.ExtraTreesRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor "sklearn.ensemble.ExtraTreesRegressor")（[n_estimators，...]） | 一棵树外的回归者。 |
| [`ensemble.GradientBoostingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier "sklearn.ensemble.GradientBoostingClassifier")([loss, …]) | Gradient Boosting用于分类。 |
| [`ensemble.GradientBoostingRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor "sklearn.ensemble.GradientBoostingRegressor")([loss, …]) | 渐变提升回归。 |
| [`ensemble.IsolationForest`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest "sklearn.ensemble.IsolationForest")（[n_estimators，...]） | 隔离森林算法 |
| [`ensemble.RandomForestClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier "sklearn.ensemble.RandomForestClassifier")（[...]） | 随机森林分类器。 |
| [`ensemble.RandomForestRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor "sklearn.ensemble.RandomForestRegressor")（[...]） | 一个随机的森林回归者。 |
| [`ensemble.RandomTreesEmbedding`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html#sklearn.ensemble.RandomTreesEmbedding "sklearn.ensemble.RandomTreesEmbedding")（[...]） | 一群完全随机的树木。 |
| [`ensemble.VotingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier "sklearn.ensemble.VotingClassifier")（estimators [，...]） | 用于不合适估计器的软投票/多数规则分类器。 |

| 类型 | 说明 |
|:--|:--|

### 部分依赖[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble.partial_dependence "永久链接到这个标题")

树集合的部分依赖图。

| 类型 | 说明 |
|:--|:--|
| [`ensemble.partial_dependence.partial_dependence`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.partial_dependence.partial_dependence.html#sklearn.ensemble.partial_dependence.partial_dependence "sklearn.ensemble.partial_dependence.partial_dependence")（......） | 部分依赖`target_variables`。 |
| [`ensemble.partial_dependence.plot_partial_dependence`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.partial_dependence.plot_partial_dependence.html#sklearn.ensemble.partial_dependence.plot_partial_dependence "sklearn.ensemble.partial_dependence.plot_partial_dependence")（......） | 部分依赖图`features`。 |

## [`sklearn.exceptions`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.exceptions "sklearn.exceptions")：异常和警告[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.exceptions "永久链接到这个标题")

该[`sklearn.exceptions`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.exceptions "sklearn.exceptions")模块包括scikit-learn中使用的所有自定义警告和错误类。

| 类型 | 说明 |
|:--|:--|
| [`exceptions.ChangedBehaviorWarning`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ChangedBehaviorWarning.html#sklearn.exceptions.ChangedBehaviorWarning "sklearn.exceptions.ChangedBehaviorWarning") | 警告类用于通知用户行为的任何更改。 |
| [`exceptions.ConvergenceWarning`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ConvergenceWarning.html#sklearn.exceptions.ConvergenceWarning "sklearn.exceptions.ConvergenceWarning") | 自定义警告以捕获收敛问题 |
| [`exceptions.DataConversionWarning`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.DataConversionWarning.html#sklearn.exceptions.DataConversionWarning "sklearn.exceptions.DataConversionWarning") | 警告用于通知代码中发生的隐式数据转换。 |
| [`exceptions.DataDimensionalityWarning`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.DataDimensionalityWarning.html#sklearn.exceptions.DataDimensionalityWarning "sklearn.exceptions.DataDimensionalityWarning") | 自定义警告，用于通知数据维度的潜在问题。 |
| [`exceptions.EfficiencyWarning`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.EfficiencyWarning.html#sklearn.exceptions.EfficiencyWarning "sklearn.exceptions.EfficiencyWarning") | 警告用于通知用户计算效率低下。 |
| [`exceptions.FitFailedWarning`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.FitFailedWarning.html#sklearn.exceptions.FitFailedWarning "sklearn.exceptions.FitFailedWarning") | 如果在拟合估计器时出错，则使用警告类。 |
| [`exceptions.NotFittedError`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html#sklearn.exceptions.NotFittedError "sklearn.exceptions.NotFittedError") | 如果在拟合之前使用估计器，则引发异常类。 |
| [`exceptions.NonBLASDotWarning`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NonBLASDotWarning.html#sklearn.exceptions.NonBLASDotWarning "sklearn.exceptions.NonBLASDotWarning") | 点操作不使用BLAS时使用的警告。 |
| [`exceptions.UndefinedMetricWarning`](http://scikit-learn.org/stable/modules/generated/sklearn.exceptions.UndefinedMetricWarning.html#sklearn.exceptions.UndefinedMetricWarning "sklearn.exceptions.UndefinedMetricWarning") | 度量标准无效时使用的警告 |

## [`sklearn.feature_extraction`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction "sklearn.feature_extraction")：特征提取[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction "永久链接到这个标题")

该[`sklearn.feature_extraction`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction "sklearn.feature_extraction")模块处理原始数据的特征提取。它目前包括从文本和图像中提取特征的方法。

**用户指南：**有关详细信息，请参阅[功能提取](http://scikit-learn.org/stable/modules/feature_extraction.html#feature-extraction)部分。

| 类型 | 说明 |
|:--|:--|
| [`feature_extraction.DictVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer "sklearn.feature_extraction.DictVectorizer")（[dtype，...]） | 将特征值映射列表转换为向量。 |
| [`feature_extraction.FeatureHasher`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher "sklearn.feature_extraction.FeatureHasher")（[...]） | 实现功能散列，即哈希技巧。 |

### 从图像[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.image "永久链接到这个标题")

该[`sklearn.feature_extraction.image`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.image "sklearn.feature_extraction.image")子模块收集实用程序从图像中提取特征。

| 类型 | 说明 |
|:--|:--|
| [`feature_extraction.image.extract_patches_2d`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html#sklearn.feature_extraction.image.extract_patches_2d "sklearn.feature_extraction.image.extract_patches_2d")（......） | 将2D图像重塑为一组补丁 |
| [`feature_extraction.image.grid_to_graph`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.grid_to_graph.html#sklearn.feature_extraction.image.grid_to_graph "sklearn.feature_extraction.image.grid_to_graph")（n_x，n_y） | 像素到像素连接的图表 |
| [`feature_extraction.image.img_to_graph`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.img_to_graph.html#sklearn.feature_extraction.image.img_to_graph "sklearn.feature_extraction.image.img_to_graph")（img [，...]） | 像素到像素梯度连接的图形 |
| [`feature_extraction.image.reconstruct_from_patches_2d`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.reconstruct_from_patches_2d.html#sklearn.feature_extraction.image.reconstruct_from_patches_2d "sklearn.feature_extraction.image.reconstruct_from_patches_2d")（......） | 从其所有补丁重建图像。 |
| [`feature_extraction.image.PatchExtractor`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html#sklearn.feature_extraction.image.PatchExtractor "sklearn.feature_extraction.image.PatchExtractor")（[...]） | 从一组图像中提取补丁 |

### 从文字[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text "永久链接到这个标题")

该[`sklearn.feature_extraction.text`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text "sklearn.feature_extraction.text")子模块收集实用程序从文本文档建立特征向量。

| 类型 | 说明 |
|:--|:--|
| [`feature_extraction.text.CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer "sklearn.feature_extraction.text.CountVectorizer")（[...]） | 将文本文档集合转换为令牌计数矩阵 |
| [`feature_extraction.text.HashingVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer "sklearn.feature_extraction.text.HashingVectorizer")（[...]） | 将文本文档集合转换为令牌出现的矩阵 |
| [`feature_extraction.text.TfidfTransformer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer "sklearn.feature_extraction.text.TfidfTransformer")（[...]） | 将计数矩阵转换为标准化的tf或tf-idf表示 |
| [`feature_extraction.text.TfidfVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer "sklearn.feature_extraction.text.TfidfVectorizer")（[...]） | 将原始文档集合转换为TF-IDF特征矩阵。 |

## [`sklearn.feature_selection`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection "sklearn.feature_selection")：特征选择[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection "永久链接到这个标题")

该[`sklearn.feature_selection`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection "sklearn.feature_selection")模块实现了特征选择算法。它目前包括单变量滤波器选择方法和递归特征消除算法。

**用户指南：**有关详细信息，请参阅[功能选择](http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)部分。

| 类型 | 说明 |
|:--|:--|
| [`feature_selection.GenericUnivariateSelect`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect "sklearn.feature_selection.GenericUnivariateSelect")（[...]） | 具有可配置策略的单变量特征选择器。 |
| [`feature_selection.SelectPercentile`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile "sklearn.feature_selection.SelectPercentile")（[...]） | 根据最高分的百分位数选择要素。 |
| [`feature_selection.SelectKBest`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest "sklearn.feature_selection.SelectKBest")（[score_func，k]） | 根据k个最高分选择功能。 |
| [`feature_selection.SelectFpr`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr "sklearn.feature_selection.SelectFpr")（[score_func，alpha]） | 过滤：根据FPR测试选择低于alpha的pvalues。 |
| [`feature_selection.SelectFdr`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html#sklearn.feature_selection.SelectFdr "sklearn.feature_selection.SelectFdr")（[score_func，alpha]） | 过滤：选择估计的错误发现率的p值 |
| [`feature_selection.SelectFromModel`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel "sklearn.feature_selection.SelectFromModel")(estimator) | 元变换器，用于根据重要性权重选择特征。 |
| [`feature_selection.SelectFwe`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html#sklearn.feature_selection.SelectFwe "sklearn.feature_selection.SelectFwe")（[score_func，alpha]） | 过滤：选择与家庭错误率对应的p值 |
| [`feature_selection.RFE`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE "sklearn.feature_selection.RFE")（estimator [，...]） | 具有递归特征消除的特征排名。 |
| [`feature_selection.RFECV`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV "sklearn.feature_selection.RFECV")（estimator [，step，...]） | 功能排名具有递归功能消除和交叉验证选择最佳功能。 |
| [`feature_selection.VarianceThreshold`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold "sklearn.feature_selection.VarianceThreshold")([threshold])	| 删除所有低方差特征的特征选择器。 |

| 类型 | 说明 |
|:--|:--|
| [`feature_selection.chi2`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2 "sklearn.feature_selection.chi2")（X，y） | 计算每个非负特征和类之间的卡方统计量。 |
| [`feature_selection.f_classif`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif "sklearn.feature_selection.f_classif")（X，y） | 计算所提供样品的ANOVA F值。 |
| [`feature_selection.f_regression`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression "sklearn.feature_selection.f_regression")（X，y [，center]） | 单变量线性回归测试。 |
| [`feature_selection.mutual_info_classif`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif "sklearn.feature_selection.mutual_info_classif")（X，y） | 估计离散目标变量的互信息。 |
| [`feature_selection.mutual_info_regression`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression "sklearn.feature_selection.mutual_info_regression")（X，y） | 估计连续目标变量的互信息。 |

## [`sklearn.gaussian_process`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process "sklearn.gaussian_process")：高斯过程[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process "永久链接到这个标题")

该[`sklearn.gaussian_process`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process "sklearn.gaussian_process")模块实现了基于高斯过程的回归和分类。

**用户指南：**有关详细信息，请参阅[高斯过程](http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process)部分。

| 类型 | 说明 |
|:--|:--|
| [`gaussian_process.GaussianProcessClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier "sklearn.gaussian_process.GaussianProcessClassifier")（[...]） | 基于拉普拉斯近似的高斯过程分类（GPC）。 |
| [`gaussian_process.GaussianProcessRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor "sklearn.gaussian_process.GaussianProcessRegressor")（[...]） | 高斯过程回归（GPR）。 |

核函数：

| 类型 | 说明 |
|:--|:--|
| [`gaussian_process.kernels.CompoundKernel`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.CompoundKernel.html#sklearn.gaussian_process.kernels.CompoundKernel "sklearn.gaussian_process.kernels.CompoundKernel")(kernels) | 内核由一组其他内核组成。 |
| [`gaussian_process.kernels.ConstantKernel`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ConstantKernel.html#sklearn.gaussian_process.kernels.ConstantKernel "sklearn.gaussian_process.kernels.ConstantKernel")（[...]） | 恒定内核。 |
| [`gaussian_process.kernels.DotProduct`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.DotProduct.html#sklearn.gaussian_process.kernels.DotProduct "sklearn.gaussian_process.kernels.DotProduct")（[...]） | 点 - 产品内核。 |
| [`gaussian_process.kernels.ExpSineSquared`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ExpSineSquared.html#sklearn.gaussian_process.kernels.ExpSineSquared "sklearn.gaussian_process.kernels.ExpSineSquared")（[...]） | Exp-Sine-Squared内核。 |
| [`gaussian_process.kernels.Exponentiation`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Exponentiation.html#sklearn.gaussian_process.kernels.Exponentiation "sklearn.gaussian_process.kernels.Exponentiation")（......） | 通过给定指数指数内核。 |
| [`gaussian_process.kernels.Hyperparameter`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Hyperparameter.html#sklearn.gaussian_process.kernels.Hyperparameter "sklearn.gaussian_process.kernels.Hyperparameter") | 内核超参数的规范以namedtuple的形式出现。 |
| [`gaussian_process.kernels.Kernel`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Kernel.html#sklearn.gaussian_process.kernels.Kernel "sklearn.gaussian_process.kernels.Kernel") | 所有内核的基类。 |
| [`gaussian_process.kernels.Matern`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html#sklearn.gaussian_process.kernels.Matern "sklearn.gaussian_process.kernels.Matern")（[...]） | Matern内核。 |
| [`gaussian_process.kernels.PairwiseKernel`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.PairwiseKernel.html#sklearn.gaussian_process.kernels.PairwiseKernel "sklearn.gaussian_process.kernels.PairwiseKernel")（[...]） | sklearn.metrics.pairwise中内核的包装器。 |
| [`gaussian_process.kernels.Product`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Product.html#sklearn.gaussian_process.kernels.Product "sklearn.gaussian_process.kernels.Product")（k1，k2） | 两个内核k1和k2的产品核k1 * k2。 |
| [`gaussian_process.kernels.RBF`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF "sklearn.gaussian_process.kernels.RBF")（[length_scale，...]） | 径向基函数内核（又称平方指数内核）。 |
| [`gaussian_process.kernels.RationalQuadratic`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html#sklearn.gaussian_process.kernels.RationalQuadratic "sklearn.gaussian_process.kernels.RationalQuadratic")（[...]） | Rational二次内核。 |
| [`gaussian_process.kernels.Sum`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Sum.html#sklearn.gaussian_process.kernels.Sum "sklearn.gaussian_process.kernels.Sum")（k1，k2） | 两个内核k1和k2的Sum-kernel k1 + k2。 |
| [`gaussian_process.kernels.WhiteKernel`](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html#sklearn.gaussian_process.kernels.WhiteKernel "sklearn.gaussian_process.kernels.WhiteKernel")（[...]） | 白仁。 |

## [`sklearn.isotonic`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic "sklearn.isotonic")：[保序回归](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic "永久链接到这个标题")

**用户指南：**有关详细信息，请参阅[Isotonic回归](http://scikit-learn.org/stable/modules/isotonic.html#isotonic)部分。

| 类型 | 说明 |
|:--|:--|
| [`isotonic.IsotonicRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression "sklearn.isotonic.IsotonicRegression")（[y_min，y_max，...]） | 保序回归模型。 |

| 类型 | 说明 |
|:--|:--|
| [`isotonic.check_increasing`](http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.check_increasing.html#sklearn.isotonic.check_increasing "sklearn.isotonic.check_increasing")（x，y） | 确定y是否与x单调相关。 |
| [`isotonic.isotonic_regression`](http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.isotonic_regression.html#sklearn.isotonic.isotonic_regression "sklearn.isotonic.isotonic_regression")（y [，...]） | 解决保序回归模型： |

## [`sklearn.impute`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute "sklearn.impute") [归为(处理缺失值)](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute "永久链接到这个标题")

变换器的缺失估算

**用户指南：**有关详细信息，请参阅 [“缺失值](http://scikit-learn.org/stable/modules/impute.html#impute)的[插补”](http://scikit-learn.org/stable/modules/impute.html#impute)部分。

| 类型 | 说明 |
|:--|:--|
| [`impute.SimpleImputer`](http://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer "sklearn.impute.SimpleImputer")（[missing_values，...]） | 用于完成缺失值的插补变换器。 |
| [`impute.MissingIndicator`](http://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html#sklearn.impute.MissingIndicator "sklearn.impute.MissingIndicator")（[missing_values，...]） | 缺失值的二进制指标。 |

## [`sklearn.kernel_approximation`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation "sklearn.kernel_approximation")[内核逼近](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation "永久链接到这个标题")

该[`sklearn.kernel_approximation`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation "sklearn.kernel_approximation")模块基于傅立叶变换实现了几个近似的内核特征映射。

**用户指南：**有关详细信息，请参阅 [“内核逼近”](http://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation)部分。

| 类型 | 说明 |
|:--|:--|
| [`kernel_approximation.AdditiveChi2Sampler`](http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler.html#sklearn.kernel_approximation.AdditiveChi2Sampler "sklearn.kernel_approximation.AdditiveChi2Sampler")（[...]） | 加性chi2核的近似特征映射。 |
| [`kernel_approximation.Nystroem`](http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem "sklearn.kernel_approximation.Nystroem")([kernel, …]) | 使用训练数据的子集近似内核映射。 |
| [`kernel_approximation.RBFSampler`](http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler "sklearn.kernel_approximation.RBFSampler")（[gamma，...]） | 通过蒙特卡罗近似的傅里叶变换逼近RBF核的特征映射。 |
| [`kernel_approximation.SkewedChi2Sampler`](http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler.html#sklearn.kernel_approximation.SkewedChi2Sampler "sklearn.kernel_approximation.SkewedChi2Sampler")（[...]） | 通过蒙特卡罗近似的傅立叶变换近似“skewed chi-squared”核的特征图。 |

## [`sklearn.kernel_ridge`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge "sklearn.kernel_ridge")[内核岭回归](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge "永久链接到这个标题")

模块[`sklearn.kernel_ridge`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge "sklearn.kernel_ridge")实现内核岭回归。

**用户指南：**有关详细信息，请参阅[内核岭回归](http://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge)部分。

| 类型 | 说明 |
|:--|:--|
| [`kernel_ridge.KernelRidge`](http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge "sklearn.kernel_ridge.KernelRidge")（[alpha，kernel，...]） | 内核岭回归。 |

## [`sklearn.linear_model`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model "sklearn.linear_model")：广义线性模型[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model "永久链接到这个标题")

该[`sklearn.linear_model`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model "sklearn.linear_model")模块实现了广义线性模型。它包括利用最小角度回归和坐标下降计算的岭回归，贝叶斯回归，套索和弹性网估计。它还实现了Stochastic Gradient Descent相关算法。

**用户指南：**有关详细信息，请参阅 [“广义线性模型”](http://scikit-learn.org/stable/modules/linear_model.html#linear-model)部分。

| 类型 | 说明 |
|:--|:--|
| [`linear_model.ARDRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression "sklearn.linear_model.ARDRegression")（[n_iter，tol，...]） | 贝叶斯ARD回归。 |
| [`linear_model.BayesianRidge`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge "sklearn.linear_model.BayesianRidge")（[n_iter，tol，...]） | 贝叶斯岭回归 |
| [`linear_model.ElasticNet`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet "sklearn.linear_model.ElasticNet")（[alpha，l1_ratio，...]） | 将L1和L2组合作为正则化器的线性回归。 |
| [`linear_model.ElasticNetCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV "sklearn.linear_model.ElasticNetCV")（[l1_ratio，eps，...]） | 具有沿正则化路径的迭代拟合的弹性网络模型 |
| [`linear_model.HuberRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor "sklearn.linear_model.HuberRegressor")（[epsilon，...]） | 线性回归模型对异常值具有鲁棒性（健壮性）。 |
| [`linear_model.Lars`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html#sklearn.linear_model.Lars "sklearn.linear_model.Lars")（[fit_intercept，verbose，...]） | 最小角度回归模型又名 |
| [`linear_model.LarsCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html#sklearn.linear_model.LarsCV "sklearn.linear_model.LarsCV")（[fit_intercept，...]） | 交叉验证的最小角度回归模型 |
| [`linear_model.Lasso`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso "sklearn.linear_model.Lasso")（[alpha，fit_intercept，...]） | 使用L1作为正则化器（也称为Lasso）训练的线性模型 |
| [`linear_model.LassoCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV "sklearn.linear_model.LassoCV")（[eps，n_alphas，...]） | 具有沿正则化路径的迭代拟合的套索线性模型 |
| [`linear_model.LassoLars`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars "sklearn.linear_model.LassoLars")（[α， …]） | 套索模型适合最小角度回归aka |
| [`linear_model.LassoLarsCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV "sklearn.linear_model.LassoLarsCV")（[fit_intercept，...]） | 交叉验证的Lasso，使用LARS算法 |
| [`linear_model.LassoLarsIC`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC "sklearn.linear_model.LassoLarsIC")([criterion, …]) | Lasso模型适合Lars使用BIC或AIC进行模型选择 |
| [`linear_model.LinearRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "sklearn.linear_model.LinearRegression")（[...]） | 普通最小二乘线性回归。 |
| [`linear_model.LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression "sklearn.linear_model.LogisticRegression")([penalty, …]) | Logistic回归（aka logit，MaxEnt）分类器。 |
| [`linear_model.LogisticRegressionCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV "sklearn.linear_model.LogisticRegressionCV")（[Cs，...]） | Logistic回归CV（aka logit，MaxEnt）分类器。 |
| [`linear_model.MultiTaskLasso`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso "sklearn.linear_model.MultiTaskLasso")（[α， …]） | 使用L1 / L2混合范数作为正则化器训练的多任务Lasso模型 |
| [`linear_model.MultiTaskElasticNet`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html#sklearn.linear_model.MultiTaskElasticNet "sklearn.linear_model.MultiTaskElasticNet")（[α， …]） | 使用L1 / L2混合范数作为正则化器训练的多任务ElasticNet模型 |
| [`linear_model.MultiTaskLassoCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html#sklearn.linear_model.MultiTaskLassoCV "sklearn.linear_model.MultiTaskLassoCV")（[eps，...]） | 具有内置交叉验证的多任务L1 / L2套索。 |
| [`linear_model.MultiTaskElasticNetCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html#sklearn.linear_model.MultiTaskElasticNetCV "sklearn.linear_model.MultiTaskElasticNetCV")（[...]） | 具有内置交叉验证的多任务L1 / L2 ElasticNet。 |
| [`linear_model.OrthogonalMatchingPursuit`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit "sklearn.linear_model.OrthogonalMatchingPursuit")（[...]） | 正交匹配追踪模型（OMP） |
| [`linear_model.OrthogonalMatchingPursuitCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV "sklearn.linear_model.OrthogonalMatchingPursuitCV")（[...]） | 交叉验证的正交匹配追踪模型（OMP） |
| [`linear_model.PassiveAggressiveClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier "sklearn.linear_model.PassiveAggressiveClassifier")（[...]） | 被动攻击分类器 |
| [`linear_model.PassiveAggressiveRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor "sklearn.linear_model.PassiveAggressiveRegressor")（[C， …]） | 被动攻击回归 |
| [`linear_model.Perceptron`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron "sklearn.linear_model.Perceptron")([penalty, alpha, …]) | 阅读[用户指南中的更多内容](http://scikit-learn.org/stable/modules/linear_model.html#perceptron)。 |
| [`linear_model.RANSACRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor "sklearn.linear_model.RANSACRegressor")（[...]） | RANSAC（RANdom SAmple Consensus）算法。 |
| [`linear_model.Ridge`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge "sklearn.linear_model.Ridge")（[alpha，fit_intercept，...]） | 具有l2正则化的线性最小二乘法。 |
| [`linear_model.RidgeClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier "sklearn.linear_model.RidgeClassifier")（[α， …]） | 使用岭回归的分类器。 |
| [`linear_model.RidgeClassifierCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV "sklearn.linear_model.RidgeClassifierCV")（[alphas，...]） | Ridge分类器，内置交叉验证。 |
| [`linear_model.RidgeCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV "sklearn.linear_model.RidgeCV")（[alphas，...]） | 具有内置交叉验证的岭回归。 |
| [`linear_model.SGDClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier "sklearn.linear_model.SGDClassifier")([loss, penalty, …]) | 具有SGD训练的线性分类器（SVM，逻辑回归，ao）。 |
| [`linear_model.SGDRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor "sklearn.linear_model.SGDRegressor")([loss, penalty, …]) | 通过最小化SGD的正则化经验损失来拟合线性模型 |
| [`linear_model.TheilSenRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor "sklearn.linear_model.TheilSenRegressor")（[...]） | Theil-Sen Estimator：稳健的多元回归模型。 |

| 类型 | 说明 |
|:--|:--|
| [`linear_model.enet_path`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.enet_path.html#sklearn.linear_model.enet_path "sklearn.linear_model.enet_path")（X，y [，l1_ratio，...]） | 用坐标下降计算弹性网路径 |
| [`linear_model.lars_path`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path "sklearn.linear_model.lars_path")（X，y [，Xy，Gram，...]） | 使用LARS算法计算最小角度回归或套索路径[1] |
| [`linear_model.lasso_path`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html#sklearn.linear_model.lasso_path "sklearn.linear_model.lasso_path")（X，y [，eps，...]） | 用坐标下降计算Lasso路径 |
| [`linear_model.logistic_regression_path`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.logistic_regression_path.html#sklearn.linear_model.logistic_regression_path "sklearn.linear_model.logistic_regression_path")（X，y） | 计算Logistic回归模型以获得正则化参数列表。 |
| [`linear_model.orthogonal_mp`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html#sklearn.linear_model.orthogonal_mp "sklearn.linear_model.orthogonal_mp")（X，y [，...]） | 正交匹配追踪（OMP） |
| [`linear_model.orthogonal_mp_gram`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp_gram.html#sklearn.linear_model.orthogonal_mp_gram "sklearn.linear_model.orthogonal_mp_gram")（Gram，Xy [，...]） | 革命正交匹配追踪（OMP） |
| [`linear_model.ridge_regression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html#sklearn.linear_model.ridge_regression "sklearn.linear_model.ridge_regression")（X，y，alpha [，...]） | 用正规方程法求解脊方程。 |

## [`sklearn.manifold`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold "sklearn.manifold")：流形学习[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold "永久链接到这个标题")

该[`sklearn.manifold`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold "sklearn.manifold")模块实现了数据嵌入技术。

**用户指南：**有关详细信息，请参阅 [“流形学习”](http://scikit-learn.org/stable/modules/manifold.html#manifold)部分。

| 类型 | 说明 |
|:--|:--|
| [`manifold.Isomap`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap "sklearn.manifold.Isomap")（[n_neighbors，n_components，...]） | Isomap嵌入 |
| [`manifold.LocallyLinearEmbedding`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding "sklearn.manifold.LocallyLinearEmbedding")（[...]） | 局部线性嵌入 |
| [`manifold.MDS`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS "sklearn.manifold.MDS")（[n_components，metric，n_init，...]） | 多维缩放 |
| [`manifold.SpectralEmbedding`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html#sklearn.manifold.SpectralEmbedding "sklearn.manifold.SpectralEmbedding")（[n_components，...]） | 用于非线性降维的光谱嵌入。 |
| [`manifold.TSNE`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE "sklearn.manifold.TSNE")（[n_components，perplexity，...]） | t分布式随机邻域嵌入。 |

| 类型 | 说明 |
|:--|:--|
| [`manifold.locally_linear_embedding`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html#sklearn.manifold.locally_linear_embedding "sklearn.manifold.locally_linear_embedding")（X， …[， …]） | 对数据执行局部线性嵌入分析。 |
| [`manifold.smacof`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html#sklearn.manifold.smacof "sklearn.manifold.smacof")(dissimilarities[, metric, …]) | 使用SMACOF算法计算多维缩放。 |
| [`manifold.spectral_embedding`](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.spectral_embedding.html#sklearn.manifold.spectral_embedding "sklearn.manifold.spectral_embedding")(adjacency[, …]) | 将样本投影到图拉普拉斯算子的第一个特征向量上。 |

## [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics "sklearn.metrics")：指标[](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics "永久链接到这个标题")

参阅 [模型评估: 量化预测的质量](http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation) 和 [成对的矩阵, 类别和核函数](http://scikit-learn.org/stable/modules/metrics.html#metrics) 章节的用户指南获取更多信息。

该[`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics "sklearn.metrics")模块包括评分函数，性能指标和成对指标以及距离计算。

### 型号选择界面[](http://scikit-learn.org/stable/modules/classes.html#model-selection-interface "永久链接到这个标题")

有关更多详细信息，请参阅用户指南[的评分参数：定义模型评估规则](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)部分。

| 类型 | 说明 |
|:--|:--|
| [`metrics.check_scoring`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.check_scoring.html#sklearn.metrics.check_scoring "sklearn.metrics.check_scoring")(estimator[, scoring, …]) | 从用户选项中确定评分器。 |
| [`metrics.get_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html#sklearn.metrics.get_scorer "sklearn.metrics.get_scorer")(scoring) | 从字符串中获得一名评分器 |
| [`metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer "sklearn.metrics.make_scorer")（score_func [，...]） | 从性能指标或损失函数中创建一个评分器。 |

### 分类指标[](http://scikit-learn.org/stable/modules/classes.html#classification-metrics "永久链接到这个标题")

有关详细信息，请参阅用户指南的 [“分类指标”](http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)部分。

| 类型 | 说明 |
|:--|:--|
| [`metrics.accuracy_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score "sklearn.metrics.accuracy_score")（y_true，y_pred [，...]） | 准确度分类得分。 |
| [`metrics.auc`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc "sklearn.metrics.auc")(x, y[, reorder]) | 使用梯形法则计算曲线下面积（AUC） |
| [`metrics.average_precision_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score "sklearn.metrics.average_precision_score")（y_true，y_score） | 根据预测分数计算平均精度（AP） |
| [`metrics.balanced_accuracy_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score "sklearn.metrics.balanced_accuracy_score")（y_true，y_pred） | 计算平衡精度 |
| [`metrics.brier_score_loss`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss "sklearn.metrics.brier_score_loss")（y_true，y_prob [，...]） | 计算Brier分数。 |
| [`metrics.classification_report`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report "sklearn.metrics.classification_report")（y_true，y_pred） | 构建显示主要分类指标的文本报告 |
| [`metrics.cohen_kappa_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score "sklearn.metrics.cohen_kappa_score")（y1，y2 [，labels，...]） | Cohen的kappa：衡量注释器间协议的统计数据。 |
| [`metrics.confusion_matrix`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix "sklearn.metrics.confusion_matrix")（y_true，y_pred [，...]） | 计算混淆矩阵以评估分类的准确性 |
| [`metrics.f1_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score "sklearn.metrics.f1_score")（y_true，y_pred [，labels，...]） | 计算F1分数，也称为平衡F分数或F分数 |
| [`metrics.fbeta_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score "sklearn.metrics.fbeta_score")（y_true，y_pred，beta [，...]） | 计算F-beta分数 |
| [`metrics.hamming_loss`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss "sklearn.metrics.hamming_loss")（y_true，y_pred [，...]） | 计算平均hamming损失。 |
| [`metrics.hinge_loss`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss "sklearn.metrics.hinge_loss")（y_true，pred_decision [，...]） | 平均铰链损失（非正则化） |
| [`metrics.jaccard_similarity_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score "sklearn.metrics.jaccard_similarity_score")（y_true，y_pred） | Jaccard相似系数得分 |
| [`metrics.log_loss`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss "sklearn.metrics.log_loss")（y_true，y_pred [，eps，...]） | 对数损失，又称逻辑损失或交叉熵损失。 |
| [`metrics.matthews_corrcoef`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef "sklearn.metrics.matthews_corrcoef")（y_true，y_pred [，...]） | 计算马修斯相关系数（MCC） |
| [`metrics.precision_recall_curve`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve "sklearn.metrics.precision_recall_curve")（y_true，......） | 计算不同概率阈值的精确调用对 |
| [`metrics.precision_recall_fscore_support`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support "sklearn.metrics.precision_recall_fscore_support")（......） | 计算每个班级的精确度，召回率率，F测量和支持 |
| [`metrics.precision_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score "sklearn.metrics.precision_score")（y_true，y_pred [，...]） | 计算精度 |
| [`metrics.recall_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score "sklearn.metrics.recall_score")（y_true，y_pred [，...]） | 计算召回率 |
| [`metrics.roc_auc_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score "sklearn.metrics.roc_auc_score")（y_true，y_score [，...]） | 根据预测分数在接收器工作特性曲线（ROC AUC）下的计算区域。 |
| [`metrics.roc_curve`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve "sklearn.metrics.roc_curve")（y_true，y_score [，...]） | 计算接收器工作特性（ROC） |
| [`metrics.zero_one_loss`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html#sklearn.metrics.zero_one_loss "sklearn.metrics.zero_one_loss")（y_true，y_pred [，...]） | 0-1分类损失。 |

### 回归指标[](http://scikit-learn.org/stable/modules/classes.html#regression-metrics "永久链接到这个标题")

有关详细信息，请参阅用户指南的回归指标”](http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)部分。

| 类型 | 说明 |
|:--|:--|
| [`metrics.explained_variance_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score "sklearn.metrics.explained_variance_score")（y_true，y_pred） | 解释方差回归分数函数 |
| [`metrics.mean_absolute_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error "sklearn.metrics.mean_absolute_error")（y_true，y_pred） | 平均绝对误差回归损失 |
| [`metrics.mean_squared_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error "sklearn.metrics.mean_squared_error")（y_true，y_pred [，...]） | 均方误差回归损失 |
| [`metrics.mean_squared_log_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error "sklearn.metrics.mean_squared_log_error")（y_true，y_pred） | 均方对数误差回归损失 |
| [`metrics.median_absolute_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error "sklearn.metrics.median_absolute_error")（y_true，y_pred） | 中位数绝对误差回归损失 |
| [`metrics.r2_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score "sklearn.metrics.r2_score")（y_true，y_pred [，...]） | R ^ 2（确定系数）回归分数函数。 |

### 多标签排名指标[](http://scikit-learn.org/stable/modules/classes.html#multilabel-ranking-metrics "永久链接到这个标题")

有关更多详细信息，请参阅用户指南的[Multilabel排名指标](http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics)部分。

| 类型 | 说明 |
|:--|:--|
| [`metrics.coverage_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.coverage_error.html#sklearn.metrics.coverage_error "sklearn.metrics.coverage_error")（y_true，y_score [，...]） | 覆盖误差测量 |
| [`metrics.label_ranking_average_precision_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html#sklearn.metrics.label_ranking_average_precision_score "sklearn.metrics.label_ranking_average_precision_score")（......） | 计算基于排名的平均精度 |
| [`metrics.label_ranking_loss`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_loss.html#sklearn.metrics.label_ranking_loss "sklearn.metrics.label_ranking_loss")（y_true，y_score） | 计算排名损失度量 |

### 聚类指标[](http://scikit-learn.org/stable/modules/classes.html#clustering-metrics "永久链接到这个标题")

有关更多详细信息，请参阅用户指南的 [“聚类性能评估”](http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)部分。

该[`sklearn.metrics.cluster`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.cluster "sklearn.metrics.cluster")子模块包含了聚类分析的结果评价指标。评估有两种形式：

*   监督，使用每个样本的基础真值类值。
*   无监督，没有和测量模型本身的“质量”。

| 类型 | 说明 |
|:--|:--|
| [`metrics.adjusted_mutual_info_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score "sklearn.metrics.adjusted_mutual_info_score")（... [，...]） | 调整两个聚类之间的相互信息。 |
| [`metrics.adjusted_rand_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score "sklearn.metrics.adjusted_rand_score")（labels_true，...） | 兰德指数调整为偶然。 |
| [`metrics.calinski_harabaz_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabaz_score.html#sklearn.metrics.calinski_harabaz_score "sklearn.metrics.calinski_harabaz_score")(X, labels) | 计算Calinski和Harabaz得分。 |
| [`metrics.davies_bouldin_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score "sklearn.metrics.davies_bouldin_score")(X, labels) | 计算戴维斯 - 布尔丁的得分。 |
| [`metrics.completeness_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score "sklearn.metrics.completeness_score")（labels_true，...） | 给出基本事实的集群标签的完整性度量。 |
| [`metrics.cluster.contingency_matrix`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html#sklearn.metrics.cluster.contingency_matrix "sklearn.metrics.cluster.contingency_matrix")（... [，...]） | 构建描述标签之间关系的应变矩阵。 |
| [`metrics.fowlkes_mallows_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score "sklearn.metrics.fowlkes_mallows_score")（labels_true，...） | 测量一组点的两个聚类的相似性。 |
| [`metrics.homogeneity_completeness_v_measure`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html#sklearn.metrics.homogeneity_completeness_v_measure "sklearn.metrics.homogeneity_completeness_v_measure")（......） | 立即计算同质性和完整性以及V-Measure分数。 |
| [`metrics.homogeneity_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score "sklearn.metrics.homogeneity_score")（labels_true，...） | 给定基础事实的聚类标记的同质性度量。 |
| [`metrics.mutual_info_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score "sklearn.metrics.mutual_info_score")（labels_true，...） | 两个聚类之间的相互信息。 |
| [`metrics.normalized_mutual_info_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score "sklearn.metrics.normalized_mutual_info_score")（... [，...]） | 两个聚类之间的归一化互信息。 |
| [`metrics.silhouette_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score "sklearn.metrics.silhouette_score")(X, labels[, …]) | 计算所有样本的平均轮廓系数。 |
| [`metrics.silhouette_samples`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html#sklearn.metrics.silhouette_samples "sklearn.metrics.silhouette_samples")(X, labels[, metric]) | 计算每个样本的Silhouette系数。 |
| [`metrics.v_measure_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score "sklearn.metrics.v_measure_score")（labels_true，labels_pred） | V-measure集群标签给出了一个基本事实。 |

### 双向度量[](http://scikit-learn.org/stable/modules/classes.html#biclustering-metrics "永久链接到这个标题")

有关详细信息，请参阅用户指南的[Biclustering评估](http://scikit-learn.org/stable/modules/biclustering.html#biclustering-evaluation)部分。

| 类型 | 说明 |
|:--|:--|
| [`metrics.consensus_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.consensus_score.html#sklearn.metrics.consensus_score "sklearn.metrics.consensus_score")(a, b[, similarity]) | 两组双向聚类的相似性。 |

### 成对指标[](http://scikit-learn.org/stable/modules/classes.html#pairwise-metrics "永久链接到这个标题")

有关更多详细信息[，](http://scikit-learn.org/stable/modules/metrics.html#metrics)请参阅用户指南的 [“成对度量标准，关联性和内核”](http://scikit-learn.org/stable/modules/metrics.html#metrics)部分。

| 类型 | 说明 |
|:--|:--|
| [`metrics.pairwise.additive_chi2_kernel`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.additive_chi2_kernel.html#sklearn.metrics.pairwise.additive_chi2_kernel "sklearn.metrics.pairwise.additive_chi2_kernel")（X [，Y]） | 计算X和Y中观察值之间的加性卡方内核 |
| [`metrics.pairwise.chi2_kernel`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.chi2_kernel.html#sklearn.metrics.pairwise.chi2_kernel "sklearn.metrics.pairwise.chi2_kernel")（X [，Y，gamma]） | 计算指数卡方内核X和Y. |
| [`metrics.pairwise.cosine_similarity`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity "sklearn.metrics.pairwise.cosine_similarity")（X [，Y，...]） | 计算X和Y中样本之间的余弦相似度。 |
| [`metrics.pairwise.cosine_distances`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html#sklearn.metrics.pairwise.cosine_distances "sklearn.metrics.pairwise.cosine_distances")（X [，Y]） | 计算X和Y中样本之间的余弦距离。 |
| [`metrics.pairwise.distance_metrics`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics "sklearn.metrics.pairwise.distance_metrics")（） | pairwise_distances的有效指标。 |
| [`metrics.pairwise.euclidean_distances`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances "sklearn.metrics.pairwise.euclidean_distances")（X [，Y，...]） | 考虑X（和Y = X）的行作为矢量，计算每对矢量之间的距离矩阵。 |
| [`metrics.pairwise.kernel_metrics`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics "sklearn.metrics.pairwise.kernel_metrics")（） | pairwise_kernels的有效指标 |
| [`metrics.pairwise.laplacian_kernel`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html#sklearn.metrics.pairwise.laplacian_kernel "sklearn.metrics.pairwise.laplacian_kernel")（X [，Y，gamma]） | 计算X和Y之间的拉普拉斯内核。 |
| [`metrics.pairwise.linear_kernel`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html#sklearn.metrics.pairwise.linear_kernel "sklearn.metrics.pairwise.linear_kernel")（X [，Y，...]） | 计算X和Y之间的线性内核。 |
| [`metrics.pairwise.manhattan_distances`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html#sklearn.metrics.pairwise.manhattan_distances "sklearn.metrics.pairwise.manhattan_distances")（X [，Y，...]） | 计算X和Y中矢量之间的L1距离。 |
| [`metrics.pairwise.pairwise_kernels`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html#sklearn.metrics.pairwise.pairwise_kernels "sklearn.metrics.pairwise.pairwise_kernels")（X [，Y，...]） | 计算数组X和可选数组Y之间的内核。 |
| [`metrics.pairwise.polynomial_kernel`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html#sklearn.metrics.pairwise.polynomial_kernel "sklearn.metrics.pairwise.polynomial_kernel")（X [，Y，...]） | 计算X和Y之间的多项式内核： |
| [`metrics.pairwise.rbf_kernel`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel "sklearn.metrics.pairwise.rbf_kernel")（X [，Y，gamma]） | 计算X和Y之间的rbf（高斯）内核： |
| [`metrics.pairwise.sigmoid_kernel`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel.html#sklearn.metrics.pairwise.sigmoid_kernel "sklearn.metrics.pairwise.sigmoid_kernel")（X [，Y，...]） | 计算X和Y之间的sigmoid内核： |
| [`metrics.pairwise.paired_euclidean_distances`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_euclidean_distances.html#sklearn.metrics.pairwise.paired_euclidean_distances "sklearn.metrics.pairwise.paired_euclidean_distances")（X，Y） | 计算X和Y之间的成对欧氏距离 |
| [`metrics.pairwise.paired_manhattan_distances`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_manhattan_distances.html#sklearn.metrics.pairwise.paired_manhattan_distances "sklearn.metrics.pairwise.paired_manhattan_distances")（X，Y） | 计算X和Y中矢量之间的L1距离。 |
| [`metrics.pairwise.paired_cosine_distances`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_cosine_distances.html#sklearn.metrics.pairwise.paired_cosine_distances "sklearn.metrics.pairwise.paired_cosine_distances")（X，Y） | 计算X和Y之间的成对余弦距离 |
| [`metrics.pairwise.paired_distances`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_distances.html#sklearn.metrics.pairwise.paired_distances "sklearn.metrics.pairwise.paired_distances")（X，Y [metric]） | 计算X和Y之间的成对距离。 |
| [`metrics.pairwise_distances`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances "sklearn.metrics.pairwise_distances")（X [，Ymetric，......]） | 从矢量数组X和可选的Y计算距离矩阵。 |
| [`metrics.pairwise_distances_argmin`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html#sklearn.metrics.pairwise_distances_argmin "sklearn.metrics.pairwise_distances_argmin")（X，Y [，...]） | 计算一个点和一组点之间的最小距离。 |
| [`metrics.pairwise_distances_argmin_min`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin_min.html#sklearn.metrics.pairwise_distances_argmin_min "sklearn.metrics.pairwise_distances_argmin_min")（X，Y） | 计算一个点和一组点之间的最小距离。 |
| [`metrics.pairwise_distances_chunked`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html#sklearn.metrics.pairwise_distances_chunked "sklearn.metrics.pairwise_distances_chunked")（X [，Y，...]） | 通过可选的缩减生成块的距离矩阵块 |

## [`sklearn.mixture`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture "sklearn.mixture")：高斯混合模型[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture "永久链接到这个标题")

该[`sklearn.mixture`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture "sklearn.mixture")模块实现了混合建模算法。

**用户指南：**有关详细信息，请参阅[高斯混合模型](http://scikit-learn.org/stable/modules/mixture.html#mixture)部分。

| 类型 | 说明 |
|:--|:--|
| [`mixture.BayesianGaussianMixture`](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture "sklearn.mixture.BayesianGaussianMixture")（[...]） | 高斯混合的变分贝叶斯估计。 |
| [`mixture.GaussianMixture`](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture "sklearn.mixture.GaussianMixture")（[n_components，...]） | 高斯混合。 |

## [`sklearn.model_selection`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection "sklearn.model_selection")：型号选择[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection "永久链接到这个标题")

**用户指南：**有关详细信息，请参阅[交叉验证：评估估计器性能](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)，[调整](http://scikit-learn.org/stable/modules/grid_search.html#grid-search)[估计器](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)[的超参数](http://scikit-learn.org/stable/modules/grid_search.html#grid-search)和 [学习曲线](http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve)部分。

### 拆分器类[](http://scikit-learn.org/stable/modules/classes.html#splitter-classes "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`model_selection.GroupKFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold "sklearn.model_selection.GroupKFold")（[n_splits]） | 具有非重叠组的K折叠迭代器变体。 |
| [`model_selection.GroupShuffleSplit`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html#sklearn.model_selection.GroupShuffleSplit "sklearn.model_selection.GroupShuffleSplit")（[...]） | Shuffle-Group（s）-Out交叉验证迭代器 |
| [`model_selection.KFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold "sklearn.model_selection.KFold")（[n_splits，shuffle，...]） | K-Folds交叉验证器 |
| [`model_selection.LeaveOneGroupOut`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut "sklearn.model_selection.LeaveOneGroupOut")（） | 保留One Group Out交叉验证器 |
| [`model_selection.LeavePGroupsOut`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html#sklearn.model_selection.LeavePGroupsOut "sklearn.model_selection.LeavePGroupsOut")（n_groups） | 让P组退出交叉验证员 |
| [`model_selection.LeaveOneOut`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut "sklearn.model_selection.LeaveOneOut")（） | Leave-One-Out交叉验证器 |
| [`model_selection.LeavePOut`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html#sklearn.model_selection.LeavePOut "sklearn.model_selection.LeavePOut")（p）的 | Leave-P-Out交叉验证器 |
| [`model_selection.PredefinedSplit`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html#sklearn.model_selection.PredefinedSplit "sklearn.model_selection.PredefinedSplit")（test_fold） | 预定义的拆分交叉验证器 |
| [`model_selection.RepeatedKFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold "sklearn.model_selection.RepeatedKFold")（[n_splits，...]） | 重复K-Fold交叉验证器。 |
| [`model_selection.RepeatedStratifiedKFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold "sklearn.model_selection.RepeatedStratifiedKFold")（[...]） | 重复分层K-fold交叉验证器。 |
| [`model_selection.ShuffleSplit`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit "sklearn.model_selection.ShuffleSplit")（[n_splits，...]） | 随机置换交叉验证器 |
| [`model_selection.StratifiedKFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold "sklearn.model_selection.StratifiedKFold")（[n_splits，...]） | 分层K-Folds交叉验证器 |
| [`model_selection.StratifiedShuffleSplit`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit "sklearn.model_selection.StratifiedShuffleSplit")（[...]） | 分层ShuffleSplit交叉验证器 |
| [`model_selection.TimeSeriesSplit`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit "sklearn.model_selection.TimeSeriesSplit")（[n_splits，...]） | 时间序列交叉验证器 |

### 拆分器功能[](http://scikit-learn.org/stable/modules/classes.html#splitter-functions "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`model_selection.check_cv`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.check_cv.html#sklearn.model_selection.check_cv "sklearn.model_selection.check_cv")（[cv，y，classifier]） | 用于构建交叉验证器的输入检查器实用程序 |
| [`model_selection.train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split "sklearn.model_selection.train_test_split")(*arrays, …) | 将数组或矩阵拆分为随机序列和测试子集 |

### 超参数优化器[](http://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`model_selection.GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV "sklearn.model_selection.GridSearchCV")(estimator, …)	 | 彻底搜索估计器的指定参数值。 |
| [`model_selection.ParameterGrid`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid "sklearn.model_selection.ParameterGrid")（param_grid） | 参数网格，每个参数网格具有离散数量的值。 |
| [`model_selection.ParameterSampler`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html#sklearn.model_selection.ParameterSampler "sklearn.model_selection.ParameterSampler")（... [，...]） | 从给定分布采样的参数生成器。 |
| [`model_selection.RandomizedSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV "sklearn.model_selection.RandomizedSearchCV")（... [，...]） | 超参数的随机搜索。 |

| 类型 | 说明 |
|:--|:--|
| [`model_selection.fit_grid_point`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.fit_grid_point.html#sklearn.model_selection.fit_grid_point "sklearn.model_selection.fit_grid_point")（X，y，...... [，...]） | 运行适合一组参数。 |

### 模型验证[](http://scikit-learn.org/stable/modules/classes.html#model-validation "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`model_selection.cross_validate`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate "sklearn.model_selection.cross_validate")(estimator, X) | 通过交叉验证评估指标，并记录适合度/得分时间。 |
| [`model_selection.cross_val_predict`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict "sklearn.model_selection.cross_val_predict")(estimator, X) | 为每个输入数据点生成交叉验证的估计值 |
| [`model_selection.cross_val_score`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score "sklearn.model_selection.cross_val_score")(estimator, X) | 通过交叉验证评估分数 |
| [`model_selection.learning_curve`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve "sklearn.model_selection.learning_curve")(estimator, X, y) | 学习曲线。 |
| [`model_selection.permutation_test_score`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html#sklearn.model_selection.permutation_test_score "sklearn.model_selection.permutation_test_score")（......） | 使用排列评估交叉验证得分的显着性 |
| [`model_selection.validation_curve`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve "sklearn.model_selection.validation_curve")(estimator, …)	 | 验证曲线。 |

## [`sklearn.multiclass`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass "sklearn.multiclass")：多类和多标签分类[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass "永久链接到这个标题")

### 多类和多标签分类策略[](http://scikit-learn.org/stable/modules/classes.html#multiclass-and-multilabel-classification-strategies "永久链接到这个标题")

<dl class="docutils" style="margin-bottom: 15px;">

<dt style="line-height: 20px; font-weight: bold;">该模块实现了多类学习算法：</dt>

<dd style="line-height: 1.5em; margin-left: 30px; margin-top: 3px; margin-bottom: 10px;">

*   one-vs-the-rest / one-vs-all
*   一VS一
*   错误纠正输出代码

</dd>

</dl>

此模块中提供的估计器是元估计器：它们需要在其构造函数中提供基本估计器。例如，可以使用这些估计器将二元分类器或回归器转换为多类分类器。也可以将这些估计器与多类估计器一起使用，以期提高其准确性或运行时性能。

scikit-learn中的所有分类器都实现了多类分类; 如果您想尝试自定义多类策略，则只需使用此模块。

one-vs-the-rest元分类器还实现了<cite style="font-style: normal;">predict_proba</cite>方法，只要这种方法由基类分类器实现即可。此方法返回单标签和多标签情况下的类成员资格的概率。请注意，在多标签情况下，概率是给定样本在给定类中下降的边际概率。因此，在多标签情况下，给定样本的所有可能标签上的这些概率的总和*将不会*总和为单位，如在单标签情况中那样。

**用户指南：**有关详细信息，请参阅[多类和多标记算法](http://scikit-learn.org/stable/modules/multiclass.html#multiclass)部分。

| 类型 | 说明 |
|:--|:--|
| [`multiclass.OneVsRestClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier "sklearn.multiclass.OneVsRestClassifier")（estimator [，...]） | One-vs-the-rest（OvR）多类/多标签策略 |
| [`multiclass.OneVsOneClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html#sklearn.multiclass.OneVsOneClassifier "sklearn.multiclass.OneVsOneClassifier")（estimator [，...]） | 一对一多类策略 |
| [`multiclass.OutputCodeClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html#sklearn.multiclass.OutputCodeClassifier "sklearn.multiclass.OutputCodeClassifier")（estimator [，...]） | （纠错）输出代码多类策略 |

## [`sklearn.multioutput`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput "sklearn.multioutput")：多输出回归和分类[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput "永久链接到这个标题")

该模块实现了多输出回归和分类。

此模块中提供的估计器是元估计器：它们需要在其构造函数中提供基本估计器。元估计器将单输出估计器扩展到多输出估计器。

**用户指南：**有关详细信息，请参阅[多类和多标记算法](http://scikit-learn.org/stable/modules/multiclass.html#multiclass)部分。

| 类型 | 说明 |
|:--|:--|
| [`multioutput.ClassifierChain`](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html#sklearn.multioutput.ClassifierChain "sklearn.multioutput.ClassifierChain")（base_estimator） | 一种多标签模型，可将二元分类器排列成链。 |
| [`multioutput.MultiOutputRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor "sklearn.multioutput.MultiOutputRegressor")(estimator) | 多目标回归 |
| [`multioutput.MultiOutputClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier "sklearn.multioutput.MultiOutputClassifier")(estimator) | 多目标分类 |
| [`multioutput.RegressorChain`](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html#sklearn.multioutput.RegressorChain "sklearn.multioutput.RegressorChain")（base_estimator [，...]） | 一种多标签模型，可将回归排列到链中。 |

## [`sklearn.naive_bayes`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes "sklearn.naive_bayes")：朴素贝叶斯[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes "永久链接到这个标题")

该[`sklearn.naive_bayes`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes "sklearn.naive_bayes")模块实现了Naive Bayes算法。这些是基于将贝叶斯定理应用于强（天真）特征独立假设的监督学习方法。

**用户指南：**有关详细信息，请参阅[朴素贝叶斯](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes)部分。

| 类型 | 说明 |
|:--|:--|
| [`naive_bayes.BernoulliNB`](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB "sklearn.naive_bayes.BernoulliNB")（[alpha，binarize，...]） | 用于多变量伯努利模型的朴素贝叶斯分类器。 |
| [`naive_bayes.GaussianNB`](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB "sklearn.naive_bayes.GaussianNB")（[priors，var_smoothing]） | 高斯朴素贝叶斯（GaussianNB） |
| [`naive_bayes.MultinomialNB`](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB "sklearn.naive_bayes.MultinomialNB")（[α， …]） | 用于多项式模型的朴素贝叶斯分类器 |
| [`naive_bayes.ComplementNB`](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB "sklearn.naive_bayes.ComplementNB")（[alpha，fit_prior，...]） | Rennie等人描述的补体朴素贝叶斯分类器。 |

## [`sklearn.neighbors`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors "sklearn.neighbors")：最近邻居[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors "永久链接到这个标题")

该[`sklearn.neighbors`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors "sklearn.neighbors")模块实现了k近邻算法。

**用户指南：**有关详细信息，请参阅[最近邻居](http://scikit-learn.org/stable/modules/neighbors.html#neighbors)部分。

| 类型 | 说明 |
|:--|:--|
| [`neighbors.BallTree`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree "sklearn.neighbors.BallTree") | BallTree用于快速广义N点问题 |
| [`neighbors.DistanceMetric`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric "sklearn.neighbors.DistanceMetric") | DistanceMetric类 |
| [`neighbors.KDTree`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree "sklearn.neighbors.KDTree") | KDTree用于快速广义N点问题 |
| [`neighbors.KernelDensity`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity "sklearn.neighbors.KernelDensity")([bandwidth, …]) | 核密度估计 |
| [`neighbors.KNeighborsClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier "sklearn.neighbors.KNeighborsClassifier")（[...]） | 实现k近邻的分类器投票。 |
| [`neighbors.KNeighborsRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor "sklearn.neighbors.KNeighborsRegressor")（[n_neighbors，...]） | 基于k-最近邻居的回归。 |
| [`neighbors.LocalOutlierFactor`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor "sklearn.neighbors.LocalOutlierFactor")（[n_neighbors，...]） | 使用局部异常因子（LOF）的无监督异常值检测 |
| [`neighbors.RadiusNeighborsClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier "sklearn.neighbors.RadiusNeighborsClassifier")（[...]） | 在给定半径内的邻居之间实施投票的分类器 |
| [`neighbors.RadiusNeighborsRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor "sklearn.neighbors.RadiusNeighborsRegressor")（[radius，...]） | 基于固定半径内的邻居的回归。 |
| [`neighbors.NearestCentroid`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid "sklearn.neighbors.NearestCentroid")([metric, …]) | 最近的质心分类器。 |
| [`neighbors.NearestNeighbors`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors "sklearn.neighbors.NearestNeighbors")（[n_neighbors，...]） | 用于实现邻居搜索的无监督学习者。 |

| 类型 | 说明 |
|:--|:--|
| [`neighbors.kneighbors_graph`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html#sklearn.neighbors.kneighbors_graph "sklearn.neighbors.kneighbors_graph")（X，n_neighbors [，...]） | 计算X中点的k-邻居的（加权）图 |
| [`neighbors.radius_neighbors_graph`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.radius_neighbors_graph.html#sklearn.neighbors.radius_neighbors_graph "sklearn.neighbors.radius_neighbors_graph")(X, radius) | 计算X中点的邻居（加权）图 |

## [`sklearn.neural_network`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network "sklearn.neural_network")：神经网络模型[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network "永久链接到这个标题")

该[`sklearn.neural_network`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network "sklearn.neural_network")模块包括基于神经网络的模型。

**用户指南：**有关详细信息，请参阅[神经网络模型（监督）](http://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised)和[神经网络模型（无监督）](http://scikit-learn.org/stable/modules/neural_networks_unsupervised.html#neural-networks-unsupervised)部分。

| 类型 | 说明 |
|:--|:--|
| [`neural_network.BernoulliRBM`](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#sklearn.neural_network.BernoulliRBM "sklearn.neural_network.BernoulliRBM")（[n_components，...]） | 伯努利限制玻尔兹曼机器（RBM）。 |
| [`neural_network.MLPClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier "sklearn.neural_network.MLPClassifier")（[...]） | 多层感知器分类器。 |
| [`neural_network.MLPRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor "sklearn.neural_network.MLPRegressor")（[...]） | 多层感知器回归器。 |

## [`sklearn.pipeline`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline "sklearn.pipeline")：管道[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline "永久链接到这个标题")

该[`sklearn.pipeline`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline "sklearn.pipeline")模块实现了用于构建复合估计器的实用程序，作为变换和估计器链。

| 类型 | 说明 |
|:--|:--|
| [`pipeline.FeatureUnion`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion "sklearn.pipeline.FeatureUnion")（transformer_list [，...]） | 连接多个变换器对象的结果。 |
| [`pipeline.Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline "sklearn.pipeline.Pipeline")(steps[, memory]) | 使用最终估计器进行变换的流水线。 |

| 类型 | 说明 |
|:--|:--|
| [`pipeline.make_pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline "sklearn.pipeline.make_pipeline")（*步骤，** kwargs） | 从给定的估计器构造管道。 |
| [`pipeline.make_union`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_union.html#sklearn.pipeline.make_union "sklearn.pipeline.make_union")(*transformers, **kwargs) | 从给定的变换器构造一个FeatureUnion。 |

## [`sklearn.preprocessing`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing "sklearn.preprocessing")：预处理和规范化[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing "永久链接到这个标题")

该[`sklearn.preprocessing`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing "sklearn.preprocessing")模块包括缩放，居中，标准化，二值化和插补方法。

**用户指南：**有关详细信息，请参阅[预处理数据](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)部分。

| 类型 | 说明 |
|:--|:--|
| [`preprocessing.Binarizer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer "sklearn.preprocessing.Binarizer")([threshold, copy]) | 根据阈值将数据二值化（将特征值设置为0或1） |
| [`preprocessing.FunctionTransformer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer "sklearn.preprocessing.FunctionTransformer")（[func，...]） | 从任意可调用构造变换器。 |
| [`preprocessing.KBinsDiscretizer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn.preprocessing.KBinsDiscretizer "sklearn.preprocessing.KBinsDiscretizer")（[n_bins，...]） | 将连续数据分成间隔。 |
| [`preprocessing.KernelCenterer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KernelCenterer.html#sklearn.preprocessing.KernelCenterer "sklearn.preprocessing.KernelCenterer")（） | 将核心矩阵居中 |
| [`preprocessing.LabelBinarizer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer "sklearn.preprocessing.LabelBinarizer")（[neg_label，...]） | 以一对一的方式对标签进行二值化 |
| [`preprocessing.LabelEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder "sklearn.preprocessing.LabelEncoder") | 编码值介于0和n_classes-1之间的标签。 |
| [`preprocessing.MultiLabelBinarizer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer "sklearn.preprocessing.MultiLabelBinarizer")（[classes，...]） | 在可迭代的迭代和多标签格式之间进行转换 |
| [`preprocessing.MaxAbsScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler "sklearn.preprocessing.MaxAbsScaler")([copy]) | 按每个特征的最大绝对值缩放。 |
| [`preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler "sklearn.preprocessing.MinMaxScaler")（[feature_range，copy]） | 通过将每个要素缩放到给定范围来转换要素。 |
| [`preprocessing.Normalizer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer "sklearn.preprocessing.Normalizer")([norm, copy]) | 将样本单独归一化为单位范数。 |
| [`preprocessing.OneHotEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder "sklearn.preprocessing.OneHotEncoder")（[n_values，...]） | 将分类整数特征编码为单热数字数组。 |
| [`preprocessing.OrdinalEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder "sklearn.preprocessing.OrdinalEncoder")（[categories，dtype]） | 将分类特征编码为整数数组。 |
| [`preprocessing.PolynomialFeatures`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures "sklearn.preprocessing.PolynomialFeatures")([degree, …]) | 生成多项式和交互功能。 |
| [`preprocessing.PowerTransformer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer "sklearn.preprocessing.PowerTransformer")([method, …]) | 应用特征功率变换使数据更像高斯。 |
| [`preprocessing.QuantileTransformer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer "sklearn.preprocessing.QuantileTransformer")（[...]） | 使用分位数信息转换要素。 |
| [`preprocessing.RobustScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler "sklearn.preprocessing.RobustScaler")（[with_centering，...]） | 使用对异常值具有鲁棒性（健壮性）的统计信息来扩展要素。 |
| [`preprocessing.StandardScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler "sklearn.preprocessing.StandardScaler")([copy, …]) | 通过删除均值和缩放到单位方差来标准化特征 |

| 类型 | 说明 |
|:--|:--|
| [`preprocessing.add_dummy_feature`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.add_dummy_feature.html#sklearn.preprocessing.add_dummy_feature "sklearn.preprocessing.add_dummy_feature")(X[, value]) | 增加具有附加虚拟特征的数据集。 |
| [`preprocessing.binarize`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.binarize.html#sklearn.preprocessing.binarize "sklearn.preprocessing.binarize")(X[, threshold, copy]) | 类数组或scipy.sparse矩阵的布尔阈值 |
| [`preprocessing.label_binarize`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html#sklearn.preprocessing.label_binarize "sklearn.preprocessing.label_binarize")(y, classes[, …]) | 以一对一的方式对标签进行二值化 |
| [`preprocessing.maxabs_scale`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html#sklearn.preprocessing.maxabs_scale "sklearn.preprocessing.maxabs_scale")(X[, axis, copy]) | 将每个要素缩放到[-1,1]范围，而不会破坏稀疏性。 |
| [`preprocessing.minmax_scale`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale "sklearn.preprocessing.minmax_scale")（X[， …]） | 通过将每个要素缩放到给定范围来转换要素。 |
| [`preprocessing.normalize`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize "sklearn.preprocessing.normalize")（X [，norm，axis，...]） | 将输入向量单独缩放到单位范数（向量长度）。 |
| [`preprocessing.quantile_transform`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform "sklearn.preprocessing.quantile_transform")（X [，轴，...]） | 使用分位数信息转换要素。 |
| [`preprocessing.robust_scale`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html#sklearn.preprocessing.robust_scale "sklearn.preprocessing.robust_scale")（X [，轴，...]） | 沿任意轴标准化数据集 |
| [`preprocessing.scale`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale "sklearn.preprocessing.scale")（X [，axis，with_mean，...]） | 沿任意轴标准化数据集 |
| [`preprocessing.power_transform`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html#sklearn.preprocessing.power_transform "sklearn.preprocessing.power_transform")（X [，方法，......]） | 应用特征功率变换使数据更像高斯。 |

## [`sklearn.random_projection`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection "sklearn.random_projection")：随机投影[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection "永久链接到这个标题")

随机投影变压器

随机投影是一种简单且计算有效的方法，通过交换受控制的精度（作为附加方差）来缩短数据的维数，从而缩短处理时间并缩小模型尺寸。

控制随机投影矩阵的尺寸和分布，以便保持数据集的任何两个样本之间的成对距离。

随机投影效率背后的主要理论结果是 [Johnson-Lindenstrauss引理（引用维基百科）](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)：

> 在数学中，Johnson-Lindenstrauss引理是关于从高维到低维欧几里德空间的低失真嵌入点的结果。该引理指出，高维空间中的一小组点可以嵌入到更低维度的空间中，使得点之间的距离几乎保持不变。用于嵌入的地图至少是Lipschitz，甚至可以被视为正交投影。

**用户指南：**有关详细信息，请参阅[随机投影](http://scikit-learn.org/stable/modules/random_projection.html#random-projection)部分。

| 类型 | 说明 |
|:--|:--|
| [`random_projection.GaussianRandomProjection`](http://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection "sklearn.random_projection.GaussianRandomProjection")（[...]） | 通过高斯随机投影降低维数 |
| [`random_projection.SparseRandomProjection`](http://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html#sklearn.random_projection.SparseRandomProjection "sklearn.random_projection.SparseRandomProjection")（[...]） | 通过稀疏随机投影减少维数 |

| 类型 | 说明 |
|:--|:--|
| [`random_projection.johnson_lindenstrauss_min_dim`](http://scikit-learn.org/stable/modules/generated/sklearn.random_projection.johnson_lindenstrauss_min_dim.html#sklearn.random_projection.johnson_lindenstrauss_min_dim "sklearn.random_projection.johnson_lindenstrauss_min_dim")（......） | 找到随机投射到的“安全”数量的组件 |

## [`sklearn.semi_supervised`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised "sklearn.semi_supervised")半监督学习[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised "永久链接到这个标题")

该[`sklearn.semi_supervised`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised "sklearn.semi_supervised")模块实现了半监督学习算法。这些算法利用少量标记数据和大量未标记数据进行分类任务。该模块包括Label Propagation。

**用户指南：**有关详细信息，请参阅[半监督](http://scikit-learn.org/stable/modules/label_propagation.html#semi-supervised)部分。

| 类型 | 说明 |
|:--|:--|
| [`semi_supervised.LabelPropagation`](http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html#sklearn.semi_supervised.LabelPropagation "sklearn.semi_supervised.LabelPropagation")([kernel, …]) | 标签传播分类器 |
| [`semi_supervised.LabelSpreading`](http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading "sklearn.semi_supervised.LabelSpreading")([kernel, …]) | LabelSpreading模型用于半监督学习 |

## [`sklearn.svm`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm "sklearn.svm")：支持向量机[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm "永久链接到这个标题")

该[`sklearn.svm`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm "sklearn.svm")模块包括支持向量机算法。

**用户指南：**有关详细信息，请参阅 [“支持向量机”](http://scikit-learn.org/stable/modules/svm.html#svm)部分。

### 估计器[](http://scikit-learn.org/stable/modules/classes.html#estimators "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`svm.LinearSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC "sklearn.svm.LinearSVC")([penalty, loss, dual, tol, C, …]) | 线性支持向量分类。 |
| [`svm.LinearSVR`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR "sklearn.svm.LinearSVR")（[epsilon，tol，C，loss，...]） | 线性支持向量回归。 |
| [`svm.NuSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC "sklearn.svm.NuSVC")（[nu，kernel，degree，gamma，...]） | Nu支持向量分类。 |
| [`svm.NuSVR`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR "sklearn.svm.NuSVR")（[nu，C，kernel，degree，gamma，...]） | Nu支持向量回归。 |
| [`svm.OneClassSVM`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM "sklearn.svm.OneClassSVM")([kernel, degree, gamma, …]) | 无监督异常值检测。 |
| [`svm.SVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC "sklearn.svm.SVC")（[C，内核，度，gamma，coef0，...]） | C-支持向量分类。 |
| [`svm.SVR`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR "sklearn.svm.SVR")（[kernel，degree，gamma，coef0，tol，...]） | Epsilon支持向量回归。 |

| 类型 | 说明 |
|:--|:--|
| [`svm.l1_min_c`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.l1_min_c.html#sklearn.svm.l1_min_c "sklearn.svm.l1_min_c")（X，y [，loss，fit_intercept，...]） | 返回C的最低边界，使得对于C in（l1_min_C，infinity），模型保证不为空。 |

### 低级方法[](http://scikit-learn.org/stable/modules/classes.html#low-level-methods "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`svm.libsvm.cross_validation`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.cross_validation.html#sklearn.svm.libsvm.cross_validation "sklearn.svm.libsvm.cross_validation") | 交叉验证程序的绑定（低级程序） |
| [`svm.libsvm.decision_function`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.decision_function.html#sklearn.svm.libsvm.decision_function "sklearn.svm.libsvm.decision_function") | 预测保证金（libsvm名称为predict_values） |
| [`svm.libsvm.fit`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.fit.html#sklearn.svm.libsvm.fit "sklearn.svm.libsvm.fit") | 使用libsvm训练模型（低级方法） |
| [`svm.libsvm.predict`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.predict.html#sklearn.svm.libsvm.predict "sklearn.svm.libsvm.predict") | 给定模型预测X的目标值（低级方法） |
| [`svm.libsvm.predict_proba`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.predict_proba.html#sklearn.svm.libsvm.predict_proba "sklearn.svm.libsvm.predict_proba") | 预测概率 |

## [`sklearn.tree`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree "sklearn.tree")：决策树[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree "永久链接到这个标题")

该[`sklearn.tree`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree "sklearn.tree")模块包括用于分类和回归的基于决策树的模型。

**用户指南：**有关详细信息，请参阅[决策树](http://scikit-learn.org/stable/modules/tree.html#tree)部分。

| 类型 | 说明 |
|:--|:--|
| [`tree.DecisionTreeClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier "sklearn.tree.DecisionTreeClassifier")([criterion, …]) | 决策树分类器。 |
| [`tree.DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor "sklearn.tree.DecisionTreeRegressor")([criterion, …]) | 决策树回归量。 |
| [`tree.ExtraTreeClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier "sklearn.tree.ExtraTreeClassifier")([criterion, …]) | 一个极随机的树分类器。 |
| [`tree.ExtraTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html#sklearn.tree.ExtraTreeRegressor "sklearn.tree.ExtraTreeRegressor")([criterion, …]) | 一个非常随机的树回归器。 |

| 类型 | 说明 |
|:--|:--|
| [`tree.export_graphviz`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz "sklearn.tree.export_graphviz")(decision_tree[, …]) | 以DOT格式导出决策树。 |

## [`sklearn.utils`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils "sklearn.utils")：实用程序[](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils "永久链接到这个标题")

该[`sklearn.utils`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils "sklearn.utils")模块包括各种实用程序。

**开发人员指南：**有关详细信息，请参阅 [“实用程序开发人员”](http://scikit-learn.org/stable/developers/utilities.html#developers-utils)页面。

| 类型 | 说明 |
|:--|:--|
| [`utils.testing.mock_mldata_urlopen`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.mock_mldata_urlopen.html#sklearn.utils.testing.mock_mldata_urlopen "sklearn.utils.testing.mock_mldata_urlopen")（* args，......） | 模拟urlopen函数的对象伪造对mldata的请求。 |

| 类型 | 说明 |
|:--|:--|
| [`utils.arrayfuncs.cholesky_delete`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.arrayfuncs.cholesky_delete.html#sklearn.utils.arrayfuncs.cholesky_delete "sklearn.utils.arrayfuncs.cholesky_delete") |  |
| [`utils.arrayfuncs.min_pos`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.arrayfuncs.min_pos.html#sklearn.utils.arrayfuncs.min_pos "sklearn.utils.arrayfuncs.min_pos") | 在正值上查找数组的最小值 |
| [`utils.as_float_array`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.as_float_array.html#sklearn.utils.as_float_array "sklearn.utils.as_float_array")（X [，copy，force_all_finite]） | 将类数组转换为浮点数组。 |
| [`utils.assert_all_finite`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.assert_all_finite.html#sklearn.utils.assert_all_finite "sklearn.utils.assert_all_finite")（X [，allow_nan]） | 如果X包含NaN或无穷大，则抛出ValueError。 |
| [`utils.bench.total_seconds`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.bench.total_seconds.html#sklearn.utils.bench.total_seconds "sklearn.utils.bench.total_seconds")(delta) | 辅助函数来模拟函数total_seconds， |
| [`utils.check_X_y`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_X_y.html#sklearn.utils.check_X_y "sklearn.utils.check_X_y")（X，y [，accept_sparse，...]） | 标准估计器的输入验证。 |
| [`utils.check_array`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html#sklearn.utils.check_array "sklearn.utils.check_array")（array [，accept_sparse，...]） | 对数组，列表，稀疏矩阵或类似的输入验证。 |
| [`utils.check_consistent_length`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_consistent_length.html#sklearn.utils.check_consistent_length "sklearn.utils.check_consistent_length")(*arrays) | 检查所有阵列是否具有一致的第一维。 |
| [`utils.check_random_state`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html#sklearn.utils.check_random_state "sklearn.utils.check_random_state")(seed) | 将种子转换为np.random.RandomState实例 |
| [`utils.class_weight.compute_class_weight`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html#sklearn.utils.class_weight.compute_class_weight "sklearn.utils.class_weight.compute_class_weight")（......） | 估算不平衡数据集的类权重。 |
| [`utils.class_weight.compute_sample_weight`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html#sklearn.utils.class_weight.compute_sample_weight "sklearn.utils.class_weight.compute_sample_weight")（......） | 对于不平衡数据集，按类别估算样本权重。 |
| [`utils.deprecated`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.deprecated.html#sklearn.utils.deprecated "sklearn.utils.deprecated")([extra]) | Decorator将函数或类标记为已弃用。 |
| [`utils.estimator_checks.check_estimator`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator "sklearn.utils.estimator_checks.check_estimator")(estimator) | 检查估算员是否遵守scikit-learn惯例。 |
| [`utils.extmath.safe_sparse_dot`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.safe_sparse_dot.html#sklearn.utils.extmath.safe_sparse_dot "sklearn.utils.extmath.safe_sparse_dot")（a，b [，...]） | 正确处理稀疏矩阵情况的点积 |
| [`utils.extmath.randomized_range_finder`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_range_finder.html#sklearn.utils.extmath.randomized_range_finder "sklearn.utils.extmath.randomized_range_finder")(A, …) | 计算一个正交矩阵，其范围近似于A的范围。 |
| [`utils.extmath.randomized_svd`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html#sklearn.utils.extmath.randomized_svd "sklearn.utils.extmath.randomized_svd")（M，n_components） | 计算截断的随机SVD |
| [`utils.extmath.fast_logdet`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.fast_logdet.html#sklearn.utils.extmath.fast_logdet "sklearn.utils.extmath.fast_logdet")(A) | 计算对数的log（det（A）） |
| [`utils.extmath.density`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.density.html#sklearn.utils.extmath.density "sklearn.utils.extmath.density")（w，** kwargs） | 计算稀疏矢量的密度 |
| [`utils.extmath.weighted_mode`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.weighted_mode.html#sklearn.utils.extmath.weighted_mode "sklearn.utils.extmath.weighted_mode")(a, w[, axis]) | 返回a中加权模态（最常见）值的数组 |
| [`utils.gen_even_slices`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.gen_even_slices.html#sklearn.utils.gen_even_slices "sklearn.utils.gen_even_slices")（n，n_packs [，n_samples]） | 生成器创建n_packs切片上升到n。 |
| [`utils.graph.single_source_shortest_path_length`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.graph.single_source_shortest_path_length.html#sklearn.utils.graph.single_source_shortest_path_length "sklearn.utils.graph.single_source_shortest_path_length")（......） | 返回从源到所有可到达节点的最短路径长度。 |
| [`utils.graph_shortest_path.graph_shortest_path`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.graph_shortest_path.graph_shortest_path.html#sklearn.utils.graph_shortest_path.graph_shortest_path "sklearn.utils.graph_shortest_path.graph_shortest_path") | 在正向或无向图上执行最短路径图搜索。 |
| [`utils.indexable`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.indexable.html#sklearn.utils.indexable "sklearn.utils.indexable")（* iterables） | 使数组可转换为交叉验证。 |
| [`utils.multiclass.type_of_target`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html#sklearn.utils.multiclass.type_of_target "sklearn.utils.multiclass.type_of_target")(y) | 确定目标指示的数据类型。 |
| [`utils.multiclass.is_multilabel`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.is_multilabel.html#sklearn.utils.multiclass.is_multilabel "sklearn.utils.multiclass.is_multilabel")(y) | 检查是否`y`采用多标签格式。 |
| [`utils.multiclass.unique_labels`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.unique_labels.html#sklearn.utils.multiclass.unique_labels "sklearn.utils.multiclass.unique_labels")(*ys) | 提取有序的唯一标签数组 |
| [`utils.murmurhash3_32`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.murmurhash3_32.html#sklearn.utils.murmurhash3_32 "sklearn.utils.murmurhash3_32") | 计算种子关键的32位杂音3。 |
| [`utils.resample`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html#sklearn.utils.resample "sklearn.utils.resample")(*arrays, **options) | 以一致的方式重新采样数组或稀疏矩阵 |
| [`utils.safe_indexing`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_indexing.html#sklearn.utils.safe_indexing "sklearn.utils.safe_indexing")(X, indices) | 使用索引从X返回项目或行。 |
| [`utils.safe_mask`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_mask.html#sklearn.utils.safe_mask "sklearn.utils.safe_mask")(X, mask) | 返回一个可以安全在X上使用的面具。 |
| [`utils.safe_sqr`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_sqr.html#sklearn.utils.safe_sqr "sklearn.utils.safe_sqr")(X[, copy]) | 元素明智的阵列喜欢和稀疏矩阵。 |
| [`utils.shuffle`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html#sklearn.utils.shuffle "sklearn.utils.shuffle")(*arrays, **options) | 以一致的方式随机播放阵列或稀疏矩阵 |
| [`utils.sparsefuncs.incr_mean_variance_axis`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.incr_mean_variance_axis.html#sklearn.utils.sparsefuncs.incr_mean_variance_axis "sklearn.utils.sparsefuncs.incr_mean_variance_axis")（X， …） | 在CSR或CSC矩阵上计算沿轴的增量均值和方差。 |
| [`utils.sparsefuncs.inplace_column_scale`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_column_scale.html#sklearn.utils.sparsefuncs.inplace_column_scale "sklearn.utils.sparsefuncs.inplace_column_scale")(X, scale) | CSC / CSR矩阵的原位列缩放。 |
| [`utils.sparsefuncs.inplace_row_scale`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_row_scale.html#sklearn.utils.sparsefuncs.inplace_row_scale "sklearn.utils.sparsefuncs.inplace_row_scale")(X, scale) | CSR或CSC矩阵的原位行缩放。 |
| [`utils.sparsefuncs.inplace_swap_row`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_swap_row.html#sklearn.utils.sparsefuncs.inplace_swap_row "sklearn.utils.sparsefuncs.inplace_swap_row")（X，m，n） | 就地交换两行CSC / CSR矩阵。 |
| [`utils.sparsefuncs.inplace_swap_column`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_swap_column.html#sklearn.utils.sparsefuncs.inplace_swap_column "sklearn.utils.sparsefuncs.inplace_swap_column")（X，m，n） | 就地交换两列CSC / CSR矩阵。 |
| [`utils.sparsefuncs.mean_variance_axis`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.mean_variance_axis.html#sklearn.utils.sparsefuncs.mean_variance_axis "sklearn.utils.sparsefuncs.mean_variance_axis")(X, axis) | 沿CSR或CSC矩阵上的轴上计算均值和方差 |
| [`utils.sparsefuncs.inplace_csr_column_scale`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_csr_column_scale.html#sklearn.utils.sparsefuncs.inplace_csr_column_scale "sklearn.utils.sparsefuncs.inplace_csr_column_scale")（X， …） | CSR矩阵的原位列缩放。 |
| [`utils.sparsefuncs_fast.inplace_csr_row_normalize_l1`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1 "sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1") | 使用l1规范对Inplace行进行规范化 |
| [`utils.sparsefuncs_fast.inplace_csr_row_normalize_l2`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2 "sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2") | 使用l2范数对Inplace行进行标准化 |
| [`utils.random.sample_without_replacement`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.random.sample_without_replacement.html#sklearn.utils.random.sample_without_replacement "sklearn.utils.random.sample_without_replacement") | 样本整数无需替换。 |
| [`utils.validation.check_is_fitted`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted "sklearn.utils.validation.check_is_fitted")(estimator, …)	 | 对估计器执行is_fitted验证。 |
| [`utils.validation.check_memory`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_memory.html#sklearn.utils.validation.check_memory "sklearn.utils.validation.check_memory")(memory) | 检查`memory`是否像joblib.Memory一样。 |
| [`utils.validation.check_symmetric`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_symmetric.html#sklearn.utils.validation.check_symmetric "sklearn.utils.validation.check_symmetric")(array[, …]) | 确保数组是2D，方形和对称。 |
| [`utils.validation.column_or_1d`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.column_or_1d.html#sklearn.utils.validation.column_or_1d "sklearn.utils.validation.column_or_1d")(y[, warn]) | Ravel列或1d numpy数组，否则会引发错误 |
| [`utils.validation.has_fit_parameter`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.has_fit_parameter.html#sklearn.utils.validation.has_fit_parameter "sklearn.utils.validation.has_fit_parameter")（......） | 检查估计器的拟合方法是否支持给定参数。 |
| [`utils.testing.assert_in`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.assert_in.html#sklearn.utils.testing.assert_in "sklearn.utils.testing.assert_in") | 就像self.assertTrue（a in b），但有一个更好的默认消息。 |
| [`utils.testing.assert_not_in`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.assert_not_in.html#sklearn.utils.testing.assert_not_in "sklearn.utils.testing.assert_not_in") | 就像self.assertTrue（不是在b中），但有一个更好的默认消息。 |
| [`utils.testing.assert_raise_message`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.assert_raise_message.html#sklearn.utils.testing.assert_raise_message "sklearn.utils.testing.assert_raise_message")（......） | 帮助函数来测试异常中引发的消息。 |
| [`utils.testing.all_estimators`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.testing.all_estimators.html#sklearn.utils.testing.all_estimators "sklearn.utils.testing.all_estimators")（[...]） | 从sklearn获取所有估计器的列表。 |

来自joblib的实用工具：

| 类型 | 说明 |
|:--|:--|
| [`utils.Memory`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.Memory.html#sklearn.utils.Memory "sklearn.utils.Memory")（[location，backend，cachedir，...]） | 一个上下文对象，用于在每次使用相同的输入参数调用函数的返回值时对其进行缓存。 |
| [`utils.Parallel`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.Parallel.html#sklearn.utils.Parallel "sklearn.utils.Parallel")（[n_jobs，backend，verbose，...]） | 用于可读并行映射的Helper类。 |

| 类型 | 说明 |
|:--|:--|
| [`utils.cpu_count`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.cpu_count.html#sklearn.utils.cpu_count "sklearn.utils.cpu_count")（） | 返回CPU数量。 |
| [`utils.delayed`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.delayed.html#sklearn.utils.delayed "sklearn.utils.delayed")(function[, check_pickle]) | Decorator用于捕获函数的参数。 |
| [`utils.parallel_backend`](http://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel_backend.html#sklearn.utils.parallel_backend "sklearn.utils.parallel_backend")(backend[, n_jobs]) | 在with块中更改Parallel使用的默认后端。 |

## 最近弃用[](http://scikit-learn.org/stable/modules/classes.html#recently-deprecated "永久链接到这个标题")

### 在0.22中删除[](http://scikit-learn.org/stable/modules/classes.html#to-be-removed-in-0-22 "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`covariance.GraphLasso`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLasso.html#sklearn.covariance.GraphLasso "sklearn.covariance.GraphLasso")（* args，** kwargs） | 利用l1惩罚估计量的稀疏逆协方差估计。 |
| [`covariance.GraphLassoCV`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLassoCV.html#sklearn.covariance.GraphLassoCV "sklearn.covariance.GraphLassoCV")（* args，** kwargs） | 稀疏逆协方差w /交叉验证的l1惩罚选择 |
| [`preprocessing.Imputer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer "sklearn.preprocessing.Imputer")（* args，** kwargs） | 用于完成缺失值的插补变换器。 |

| 类型 | 说明 |
|:--|:--|
| [`covariance.graph_lasso`](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.graph_lasso.html#sklearn.covariance.graph_lasso "sklearn.covariance.graph_lasso")（emp_cov，alpha [，...]） | 已弃用：'graph_lasso'在版本0.20中重命名为'graphical_lasso'，将在0.22中删除。 |
| [`datasets.fetch_mldata`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_mldata.html#sklearn.datasets.fetch_mldata "sklearn.datasets.fetch_mldata")（dataname [，...]） | 已弃用：fetch_mldata在版本0.20中已弃用，将在版本0.22中删除 |

### 在0.21中删除[](http://scikit-learn.org/stable/modules/classes.html#to-be-removed-in-0-21 "永久链接到这个标题")

| 类型 | 说明 |
|:--|:--|
| [`linear_model.RandomizedLasso`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLasso.html#sklearn.linear_model.RandomizedLasso "sklearn.linear_model.RandomizedLasso")（* args，** kwargs） | 随机套索。 |
| [`linear_model.RandomizedLogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLogisticRegression.html#sklearn.linear_model.RandomizedLogisticRegression "sklearn.linear_model.RandomizedLogisticRegression")（......） | 随机Logistic回归 |
| [`neighbors.LSHForest`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LSHForest.html#sklearn.neighbors.LSHForest "sklearn.neighbors.LSHForest")（[n_estimators，radius，...]） | 使用LSH林执行近似最近邻搜索。 |

| 类型 | 说明 |
|:--|:--|
| [`datasets.load_mlcomp`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_mlcomp.html#sklearn.datasets.load_mlcomp "sklearn.datasets.load_mlcomp")（name_or_id [，set_，...]） | 弃用：由于[http://mlcomp.org/](http://mlcomp.org/)网站将于2017年3月关闭，因此在0.19版本中不推荐使用load_mlcomp函数，并且将在0.21中删除该函数。 |
| [`linear_model.lasso_stability_path`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_stability_path.html#sklearn.linear_model.lasso_stability_path "sklearn.linear_model.lasso_stability_path")（X，y [，...]） | DEPRECATED：函数lasso_stability_path在0.19中已弃用，将在0.21中删除。 |

©2007 - 2018，scikit-learn developers（BSD License）。 [显示此页面来源](http://scikit-learn.org/stable/_sources/modules/classes.rst.txt)

原文：<http://scikit-learn.org/stable/modules/classes.html>