# 准备工作： 工具介绍
---

# AnaConda
---

官方网站: <https://www.anaconda.com/>

最受欢迎的Python数据科学平台

Anaconda Distribution

拥有超过600万用户，开源Anaconda Distribution是在Linux，Windows和Mac OS X上进行Python和R数据科学和机器学习的最快和最简单的方法。它是单机上开发，测试和培训的行业标准。

[Anaconda (Python发行版)](https://zh.wikipedia.org/wiki/Anaconda_(Python%E5%8F%91%E8%A1%8C%E7%89%88))

Anaconda 是一种Python语言的免费增值开源发行版，用于进行大规模数据处理、预测分析，和科学计算，致力于简化包的管理和部署。Anaconda使用软件包管理系统Conda进行包管理。

下载后直接双击安装。使用时，可以点击启动相应的编程环境：

* Python(shell) ： 标准CPython
* IPython(shell)： 相当于在命令窗口的命令提示符后输入ipython回车。`pip install ipython`安装的ipython用法一样。
* Ipython QTConsole
* IPython Notebook：直接点击打开，或者在命令提示符中输入ipython.exe notebook
* Jupyter QTConsole
* Jupyter Notebook：直接点击打开，或在终端中输入： jupyter notebook 以启动服务器；在浏览器中打开notebook页面地址：<http://localhost:8888> 。Jupyter Notebook是一种 Web 应用，能让用户将说明文本、数学方程、代码和可视化内容全部组合到一个易于共享的文档中。
* Spyder：直接点击打开IDE。最大优点就是模仿MATLAB的“工作空间”
* Anaconda Prompt ： 命令行终端
* 支持其他IDE，如Pycharm

## 安装包管理，

* 列出已经安装的包：在命令提示符中输入`pip list`或者用`conda list`
* 安装新包：在命令提示符中输入`pip install 包名`，或者`conda install 包名`
* 更新包： `conda update package_name`
* 升级所有包： `conda upgrade --all`
* 卸载包：`conda remove package_names`
* 搜索包：`conda search search_term`

## 管理环境：

* 安装nb_conda，用于notebook自动关联nb_conda的环境
* 创建环境：在Anaconda终端中 `conda create -n env_name package_names[=ver]`
* 使用环境：在Anaconda终端中 `activate env_name`
* 离开环境：在Anaconda终端中 `deactivate`
* 导出环境设置：`conda env export > environmentName.yaml 或 pip freeze > environmentName.txt`
* 导入环境设置：`conda env update -f=/path/environmentName.yaml` 或 `pip install -r /path/environmentName.txt`
* 列出环境清单：`conda env list`
* 删除环境： `conda env remove -n env_name`

# NumPy
---

官方网站: <https://www.anaconda.com/>

NumPy是使用Python进行科学计算的基础包。它包含其他内容：

*   一个强大的N维数组对象
*   复杂的（广播）功能
*   用于集成C / C ++和Fortran代码的工具
*   有用的线性代数，傅里叶变换和随机数功能

除了明显的科学用途外，NumPy还可以用作通用数据的高效多维容器。可以定义任意数据类型。这使NumPy能够无缝快速地与各种数据库集成。

NumPy根据[BSD许可证授权](http://www.numpy.org/license.html#license)，只需很少的限制即可重复使用。

## 入门[](http://www.numpy.org/#getting-started "永久链接到这个标题")

*   [获得NumPy](http://www.scipy.org/scipylib/download.html)
*   [安装SciPy堆栈](http://www.scipy.org/install.html)
*   [NumPy和SciPy文档页面](http://docs.scipy.org/doc/)
*   [NumPy教程](https://docs.scipy.org/doc/numpy/user/quickstart.html)
*   [NumPy for MATLAB©用户](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html)
*   [NumPy按类别运行](https://docs.scipy.org/doc/numpy/reference/routines.html)
*   [NumPy邮件列表](http://www.scipy.org/scipylib/mailing-lists.html)

有关SciPy Stack（NumPy提供基本数组数据结构）的更多信息，请参阅[scipy.org](http://www.scipy.org/)。

![维基百科 NumPy](https://zh.wikipedia.org/wiki/NumPy)


**NumPy**是[Python语言](https://zh.wikipedia.org/wiki/Python "Python")的一个扩展程序库。支持高阶大量的[维度](https://zh.wikipedia.org/wiki/%E5%A4%9A%E7%B6%AD "多维")[数组](https://zh.wikipedia.org/wiki/%E9%99%A3%E5%88%97 "数组")与[矩阵](https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%99%A3 "矩阵")运算，此外也针对数组运算提供大量的[数学](https://zh.wikipedia.org/wiki/%E6%95%B8%E5%AD%B8 "数学")[函数](https://zh.wikipedia.org/wiki/%E5%87%BD%E6%95%B8 "函数")[库](https://zh.wikipedia.org/wiki/%E5%87%BD%E5%BC%8F%E5%BA%AB "库")。NumPy的前身**Numeric**最早是由Jim Hugunin与其它协作者共同开发，2005年，Travis Oliphant在Numeric中结合了另一个同性质的程序库Numarray的特色，并加入了其它扩展而开发了NumPy。NumPy为开放源代码并且由许多协作者共同维护开发。

## 特色

NumPy参考[CPython](https://zh.wikipedia.org/wiki/CPython "CPython")(一个使用[字节码](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E7%A0%81 "字节码")的[解释器](https://zh.wikipedia.org/wiki/%E7%9B%B4%E8%AD%AF%E5%99%A8 "解释器"))，而在这个Python实现解释器上所写的数学[算法](https://zh.wikipedia.org/wiki/%E6%BC%94%E7%AE%97%E6%B3%95 "算法")代码通常远比[编译](https://zh.wikipedia.org/wiki/%E7%BC%96%E8%AF%91 "编译")过的相同代码要来得慢。为了解决这个难题，NumPy引入了多维数组以及可以直接有效率地操作多维数组的[函数](https://zh.wikipedia.org/wiki/%E5%87%BD%E5%BC%8F "函数")与运算符。因此在NumPy上只要能被表示为针对数组或矩阵运算的算法，其运行效率几乎都可以与编译过的等效[C语言](https://zh.wikipedia.org/wiki/C%E8%AA%9E%E8%A8%80 "C语言")代码一样快。

NumPy提供了与[MATLAB](https://zh.wikipedia.org/wiki/MATLAB "MATLAB")相似的功能与操作方式，因为两者皆为解释型语言，并且都可以让用户在针对数组或矩阵运算时提供较[标量](https://zh.wikipedia.org/wiki/%E7%B4%94%E9%87%8F "标量")运算更快的性能。两者相较之下，MATLAB提供了大量的扩展工具箱(例如[Simulink](https://zh.wikipedia.org/wiki/Simulink "Simulink"))；而NumPy则是根基于Python这个更现代、完整并且开放源代码的编程语言之上。此外NumPy也可以结合其它的Python扩展库。例如[SciPy](https://zh.wikipedia.org/wiki/SciPy "SciPy")，这个库提供了更多与MATLAB相似的功能；以及[Matplotlib](https://zh.wikipedia.org/wiki/Matplotlib "Matplotlib")，这是一个与MATLAB内置绘图功能类似的库。而从本质上来说，NumPy与MATLAB同样是利用[BLAS](https://zh.wikipedia.org/wiki/BLAS "BLAS")与[LAPACK](https://zh.wikipedia.org/wiki/LAPACK "LAPACK")来提供高效率的线性代数运算。

### ndarray 数据结构

NumPy的核心功能是"ndarray"(即*n*-dimensional array，多维数组)数据结构。这是一个表示多维度、同质并且固定大小的数组对象。而由一个与此数组相关系的数据类型对象来描述其数组元素的数据格式(例如其字符组顺序、在存储器中占用的字符组数量、整数或者浮点数等等)。

