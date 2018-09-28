# 100 numpy练习

这是在numpy邮件列表，stackoverflow和numpy文档中收集的练习集合。 该系列的目标是为新老用户提供快速参考，同时为教学人员提供一系列练习。

如果您发现错误或认为您有更好的方法来解决其中一些错误，请随时在<https://github.com/rougier/numpy-100>上打开一个issue

#### 1. 以`np`为别名导入numpy包 (★☆☆)


```python
import numpy as np
```

#### 2. 打印numpy版本和配置 (★☆☆)


```python
print(np.__version__)
np.show_config()
```

    1.14.3
    mkl_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/iosdevlog/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/iosdevlog/anaconda3/include']
    blas_mkl_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/iosdevlog/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/iosdevlog/anaconda3/include']
    blas_opt_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/iosdevlog/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/iosdevlog/anaconda3/include']
    lapack_mkl_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/iosdevlog/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/iosdevlog/anaconda3/include']
    lapack_opt_info:
        libraries = ['mkl_rt', 'pthread']
        library_dirs = ['/Users/iosdevlog/anaconda3/lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['/Users/iosdevlog/anaconda3/include']


#### 3. 创建一个大小为10的空向量 (★☆☆)


```python
Z = np.zeros(10)
print(Z)
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]


#### 4.  如何查找数组的内存大小 (★☆☆)


```python
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
```

    800 bytes


#### 5.  如何从命令行获取numpy add函数的文档? (★☆☆)


```python
%run `python -c "import numpy; numpy.info(numpy.add)"`
```

    ERROR:root:File `'`python.py'` not found.


#### 6.  创建大小为10的空向量，但第五个值为1 (★☆☆)


```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```

    [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]


#### 7.  创建一个值为10到49的向量 (★☆☆)


```python
Z = np.arange(10,50)
print(Z)
```

    [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
     34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]


#### 8.  反转向量（第一个元素成为最后一个） (★☆☆)


```python
Z = np.arange(50)
Z = Z[::-1]
print(Z)
```

    [49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26
     25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2
      1  0]


#### 9.  创建一个3x3矩阵，其值范围为0到8 (★☆☆)


```python
Z = np.arange(9).reshape(3,3)
print(Z)
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]


#### 10.从 \[1,2,0,0,4,0\] 中查找非零元素的索引 (★☆☆)


```python
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
```

    (array([0, 1, 4]),)


#### 11. 创建3x3单位矩阵 (★☆☆)


```python
Z = np.eye(3)
print(Z)
```

    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]


#### 12. 使用随机值创建3x3x3数组 (★☆☆)


```python
Z = np.random.random((3,3,3))
print(Z)
```

    [[[0.18940189 0.24401418 0.78815012]
      [0.58839657 0.10791225 0.13944297]
      [0.03846002 0.51690979 0.1773832 ]]

     [[0.10936357 0.62650535 0.88865398]
      [0.54592608 0.00891409 0.28495577]
      [0.28116829 0.65418964 0.63222084]]

     [[0.39533301 0.66701207 0.14181829]
      [0.96518132 0.65596745 0.46404387]
      [0.16210843 0.69280131 0.61307487]]]


#### 13. 创建具有随机值的10x10数组，并查找最小值和最大值 (★☆☆)


```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```

    0.005966934200455354 0.9709766992958699


#### 14. 创建一个大小为30的随机向量并找到平均值 (★☆☆)


```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```

    0.48759116047915824


#### 15. 创建一个2d数组，边框为1，内部为0 (★☆☆)


```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```

    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]


#### 16. 如何在现有数组周围添加边框（填充0）？ (★☆☆)


```python
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)
```

    [[0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 1. 1. 1. 1. 0.]
     [0. 1. 1. 1. 1. 1. 0.]
     [0. 1. 1. 1. 1. 1. 0.]
     [0. 1. 1. 1. 1. 1. 0.]
     [0. 1. 1. 1. 1. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0.]]


#### 17. 以下表达式的结果是什么？ (★☆☆)


```python
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
```

    nan
    False
    False
    nan
    False


#### 18. 在对角线下方创建一个值为1,2,3,4的5x5矩阵 (★☆☆)


```python
Z = np.diag(1+np.arange(4),k=-1)
print(Z)
```

    [[0 0 0 0 0]
     [1 0 0 0 0]
     [0 2 0 0 0]
     [0 0 3 0 0]
     [0 0 0 4 0]]


#### 19. 创建一个8x8矩阵并用棋盘图案填充它 (★☆☆)


```python
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
```

    [[0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]]


#### 20. 考虑一个（6,7,8）形状数组，第100个元素的索引（x，y，z）是什么？


```python
print(np.unravel_index(100,(6,7,8)))
```

    (1, 5, 4)


#### 21. 使用tile函数创建棋盘格8x8矩阵 (★☆☆)


```python
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)
```

    [[0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]]


#### 22. 归一化5x5随机矩阵 (★☆☆)


```python
Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)
```

    [[0.55509657 0.37587308 0.23134119 0.46575257 0.24996117]
     [0.80440277 0.64032567 0.85751704 0.7135663  0.16116293]
     [0.61651013 0.35146987 0.70590428 0.41638202 0.05633396]
     [0.17004449 0.87241967 0.97099604 0.49138827 0.70547156]
     [1.         0.04008666 0.         0.12300386 0.83911085]]


#### 23. 创建一个自定义dtype，将颜色描述为四个无符号字节（RGBA） (★☆☆)


```python
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
```

#### 24. 将5x3矩阵乘以3x2矩阵（实矩阵乘积） (★☆☆)


```python
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Alternative solution, in Python 3.5 and above
Z = np.ones((5,3)) @ np.ones((3,2))
```

    [[3. 3.]
     [3. 3.]
     [3. 3.]
     [3. 3.]
     [3. 3.]]


#### 25. 给定一维数组，否定所有在3到8之间的元素。(★☆☆)


```python
# Author: Evgeni Burovski

Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)
```

    [ 0  1  2  3 -4 -5 -6 -7 -8  9 10]


#### 26. 以下脚本的输出是什么？ (★☆☆)


```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

    9
    10


#### 27. 考虑整数向量Z，这些表达式中哪些是合法的？ (★☆☆)


```python
Z = np.zeros(10)
Z**Z
# 2 << Z >> 2 TypeError: ufunc 'left_shift' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
Z <- Z
1j*Z
Z/1/1
# Z<Z>Z ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



#### 28. 以下表达式的结果是什么？


```python
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
```

    nan
    0
    [-9.22337204e+18]


    /Users/iosdevlog/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in true_divide
      """Entry point for launching an IPython kernel.
    /Users/iosdevlog/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in floor_divide



#### 29. 如何从零浮点数舍入？ (★☆☆)


```python
# Author: Charles R Harris

Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))
```

    [10. -8.  2.  7. 10. -1. -6. -3. -5.  5.]


#### 30. 如何在两个数组之间找到常用值？ (★☆☆)


```python
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))
```

    [2 4 6 7 8]


#### 31. 如何忽略所有numpy警告（不推荐）？ (★☆☆)


```python
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

# An equivalent way, with a context manager:

with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0
```

#### 32. 以下表达式是 true 吗？(★☆☆)


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

    /Users/iosdevlog/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in sqrt
      """Entry point for launching an IPython kernel.





    False



#### 33. 如何获取昨天，今天和明天的日期？ (★☆☆)


```python
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
```

#### 34. 如何获得与2016年7月相对应的所有日期？ (★★☆)


```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
```

    ['2016-07-01' '2016-07-02' '2016-07-03' '2016-07-04' '2016-07-05'
     '2016-07-06' '2016-07-07' '2016-07-08' '2016-07-09' '2016-07-10'
     '2016-07-11' '2016-07-12' '2016-07-13' '2016-07-14' '2016-07-15'
     '2016-07-16' '2016-07-17' '2016-07-18' '2016-07-19' '2016-07-20'
     '2016-07-21' '2016-07-22' '2016-07-23' '2016-07-24' '2016-07-25'
     '2016-07-26' '2016-07-27' '2016-07-28' '2016-07-29' '2016-07-30'
     '2016-07-31']


#### 35. 如何计算((A+B)\*(-A/2)) （不copy）？(★★☆)


```python
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
```




    array([-1.5, -1.5, -1.5])



#### 36. 使用5种不同的方法提取随机数组的整数部分 (★★☆)


```python
Z = np.random.uniform(0,10,10)

print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))
```

    [2. 0. 0. 3. 9. 8. 8. 4. 8. 6.]
    [2. 0. 0. 3. 9. 8. 8. 4. 8. 6.]
    [2. 0. 0. 3. 9. 8. 8. 4. 8. 6.]
    [2 0 0 3 9 8 8 4 8 6]
    [2. 0. 0. 3. 9. 8. 8. 4. 8. 6.]


#### 37. 创建一个5x5矩阵，行值范围为0到4 (★★☆)


```python
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)
```

    [[0. 1. 2. 3. 4.]
     [0. 1. 2. 3. 4.]
     [0. 1. 2. 3. 4.]
     [0. 1. 2. 3. 4.]
     [0. 1. 2. 3. 4.]]


#### 38. 考虑生成10个整数并使用它来构建数组的生成器函数(★☆☆)


```python
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)
```

    [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]


#### 39. 创建一个大小为10的向量，其值范围为0到1，两者都被排除(★★☆)


```python
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)
```

    [0.09090909 0.18181818 0.27272727 0.36363636 0.45454545 0.54545455
     0.63636364 0.72727273 0.81818182 0.90909091]


#### 40. 创建一个大小为10的随机向量并对其进行排序(★★☆)


```python
Z = np.random.random(10)
Z.sort()
print(Z)
```

    [0.40677886 0.48781003 0.53260125 0.5618158  0.67799675 0.69726518
     0.79498446 0.8386311  0.88916091 0.92433598]


#### 41. 如何比np.sum更快地求和一个小数组？ (★★☆)


```python
# Author: Evgeni Burovski

Z = np.arange(10)
np.add.reduce(Z)
```




    45



#### 42. 考虑两个随机数组A和B，检查它们是否相等(★★☆)


```python
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)

# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A,B)
print(equal)
```

    False
    False


#### 43. 使数组不可变（只读） (★★☆)


```python
Z = np.zeros(10)
Z.flags.writeable = False
# Z[0] = 1 ValueError: assignment destination is read-only
```

#### 44. 考虑一个代表笛卡尔坐标的随机10x2矩阵，将它们转换为极坐标 (★★☆)


```python
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
```

    [0.94804189 0.99743992 0.65435914 0.47736464 0.99238114 0.79600684
     1.10987777 0.71877801 0.30159635 0.89130666]
    [1.5341378  0.74674323 0.6740663  0.30887697 1.16382935 1.14545565
     0.78846281 0.46102879 1.08771013 0.8163812 ]


#### 45. 创建大小为10的随机向量，并将最大值替换为0 (★★☆)


```python
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```

    [0.70457439 0.77863143 0.0044212  0.35199635 0.55088326 0.81023126
     0.00358274 0.         0.56607909 0.96080501]


#### 46. 创建一个带有`x`和`y`坐标的结构化数组，覆盖 \[0,1\]x\[0,1\] 区域(★★☆)


```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)
```

    [[(0.  , 0.  ) (0.25, 0.  ) (0.5 , 0.  ) (0.75, 0.  ) (1.  , 0.  )]
     [(0.  , 0.25) (0.25, 0.25) (0.5 , 0.25) (0.75, 0.25) (1.  , 0.25)]
     [(0.  , 0.5 ) (0.25, 0.5 ) (0.5 , 0.5 ) (0.75, 0.5 ) (1.  , 0.5 )]
     [(0.  , 0.75) (0.25, 0.75) (0.5 , 0.75) (0.75, 0.75) (1.  , 0.75)]
     [(0.  , 1.  ) (0.25, 1.  ) (0.5 , 1.  ) (0.75, 1.  ) (1.  , 1.  )]]


####  47. 给定两个数组X和Y，构造Cauchy矩阵C (Cij =1/(xi - yj))


```python
# Author: Evgeni Burovski

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
```

    3638.1636371179666


#### 48. 打印每个numpy标量类型的最小和最大可表示值 (★★☆)


```python
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```

    -128
    127
    -2147483648
    2147483647
    -9223372036854775808
    9223372036854775807
    -3.4028235e+38
    3.4028235e+38
    1.1920929e-07
    -1.7976931348623157e+308
    1.7976931348623157e+308
    2.220446049250313e-16


#### 49. 如何打印数组的所有值？ (★★☆)


```python
np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)
```

    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]


#### 50. 如何在向量中找到最接近的值（给定标量）？ (★★☆)


```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```

    21


#### 51. 创建表示位置（x，y）和颜色（r，g，b）的结构化数组 (★★☆)


```python
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)
```

    [((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
     ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
     ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
     ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
     ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))]


#### 52. 考虑一个随机向量，其形状（100,2）代表坐标，逐点找到 (★★☆)


```python
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)
```

    [[0.         0.46674667 0.16296951 0.44951662 0.55535765 0.7538931
      0.5485385  0.35866905 0.72387053 0.42485192]
     [0.46674667 0.         0.31210931 0.77034733 0.20120695 0.2992896
      0.2822809  0.37194373 0.2683291  0.67439901]
     [0.16296951 0.31210931 0.         0.56702915 0.43285335 0.6073399
      0.45057077 0.24777672 0.57667914 0.51127994]
     [0.44951662 0.77034733 0.56702915 0.         0.73838529 0.97994021
      0.66840648 0.80222551 0.95643301 0.12589724]
     [0.55535765 0.20120695 0.43285335 0.73838529 0.         0.24202061
      0.10153246 0.56030413 0.22043036 0.62351973]
     [0.7538931  0.2992896  0.6073399  0.97994021 0.24202061 0.
      0.32688654 0.651379   0.03096698 0.86553859]
     [0.5485385  0.2822809  0.45057077 0.66840648 0.10153246 0.32688654
      0.         0.61511685 0.3099214  0.54854237]
     [0.35866905 0.37194373 0.24777672 0.80222551 0.56030413 0.651379
      0.61511685 0.         0.62144624 0.75656655]
     [0.72387053 0.2683291  0.57667914 0.95643301 0.22043036 0.03096698
      0.3099214  0.62144624 0.         0.84326608]
     [0.42485192 0.67439901 0.51127994 0.12589724 0.62351973 0.86553859
      0.54854237 0.75656655 0.84326608 0.        ]]
    [[0.         0.40057211 0.45613881 0.42802362 0.27101803 0.29164777
      0.49951555 0.30343991 0.8059705  0.30003557]
     [0.40057211 0.         0.77454001 0.70562288 0.13952453 0.39290414
      0.87907822 0.1153288  0.82790459 0.178253  ]
     [0.45613881 0.77454001 0.         0.10063928 0.63729397 0.74057755
      0.21712083 0.71462781 0.62379172 0.60652949]
     [0.42802362 0.70562288 0.10063928 0.         0.57205672 0.71929561
      0.31173555 0.65686589 0.53671417 0.53213082]
     [0.27101803 0.13952453 0.63729397 0.57205672 0.         0.34701234
      0.73992755 0.10122001 0.75669635 0.08218758]
     [0.29164777 0.39290414 0.74057755 0.71929561 0.34701234 0.
      0.74217924 0.2832675  1.06018475 0.42405961]
     [0.49951555 0.87907822 0.21712083 0.31173555 0.73992755 0.74217924
      0.         0.79672811 0.84058782 0.73096263]
     [0.30343991 0.1153288  0.71462781 0.65686589 0.10122001 0.2832675
      0.79672811 0.         0.85674689 0.18021966]
     [0.8059705  0.82790459 0.62379172 0.53671417 0.75669635 1.06018475
      0.84058782 0.85674689 0.         0.67659915]
     [0.30003557 0.178253   0.60652949 0.53213082 0.08218758 0.42405961
      0.73096263 0.18021966 0.67659915 0.        ]]


#### 53. 如何将float（32位）数组转换为整数（32位）？


```python
Z = np.arange(10, dtype=np.float32)
Z = Z.astype(np.int32, copy=False)
print(Z)
```

    [0 1 2 3 4 5 6 7 8 9]


#### 54. 如何阅读以下文件？ (★★☆)


```python
from io import StringIO

# Fake file
s = StringIO("""1, 2, 3, 4, 5\n
                6,  ,  , 7, 8\n
                 ,  , 9,10,11\n""")
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```

    [[ 1  2  3  4  5]
     [ 6 -1 -1  7  8]
     [-1 -1  9 10 11]]


#### 55. numpy数组的枚举相当于什么？ (★★☆)


```python
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
```

    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) 3
    (1, 1) 4
    (1, 2) 5
    (2, 0) 6
    (2, 1) 7
    (2, 2) 8
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) 3
    (1, 1) 4
    (1, 2) 5
    (2, 0) 6
    (2, 1) 7
    (2, 2) 8


#### 56. 生成通用的2D高斯类数组 (★★☆)


```python
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
```

    [[0.36787944 0.44822088 0.51979489 0.57375342 0.60279818 0.60279818
      0.57375342 0.51979489 0.44822088 0.36787944]
     [0.44822088 0.54610814 0.63331324 0.69905581 0.73444367 0.73444367
      0.69905581 0.63331324 0.54610814 0.44822088]
     [0.51979489 0.63331324 0.73444367 0.81068432 0.85172308 0.85172308
      0.81068432 0.73444367 0.63331324 0.51979489]
     [0.57375342 0.69905581 0.81068432 0.89483932 0.9401382  0.9401382
      0.89483932 0.81068432 0.69905581 0.57375342]
     [0.60279818 0.73444367 0.85172308 0.9401382  0.98773022 0.98773022
      0.9401382  0.85172308 0.73444367 0.60279818]
     [0.60279818 0.73444367 0.85172308 0.9401382  0.98773022 0.98773022
      0.9401382  0.85172308 0.73444367 0.60279818]
     [0.57375342 0.69905581 0.81068432 0.89483932 0.9401382  0.9401382
      0.89483932 0.81068432 0.69905581 0.57375342]
     [0.51979489 0.63331324 0.73444367 0.81068432 0.85172308 0.85172308
      0.81068432 0.73444367 0.63331324 0.51979489]
     [0.44822088 0.54610814 0.63331324 0.69905581 0.73444367 0.73444367
      0.69905581 0.63331324 0.54610814 0.44822088]
     [0.36787944 0.44822088 0.51979489 0.57375342 0.60279818 0.60279818
      0.57375342 0.51979489 0.44822088 0.36787944]]


#### 57. 如何将p元素随机放置在2D数组中？ (★★☆)


```python
# Author: Divakar

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)
```

    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]


#### 58. 减去矩阵每行的平均值 (★★☆)


```python
# Author: Warren Weckesser

X = np.random.rand(5, 10)

# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)

# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)
```

    [[-0.06403711  0.23708949  0.1562551   0.15318601  0.28405441 -0.39551655
       0.20993944  0.12535543 -0.47667329 -0.22965293]
     [ 0.04208767 -0.10476879 -0.18270018 -0.34223998  0.41696735 -0.28702877
      -0.49975397  0.46709318  0.27105368  0.21928981]
     [ 0.16593998  0.3302678  -0.43458471 -0.26987865  0.02339813  0.0995392
      -0.35194789  0.36004351  0.14701588 -0.06979325]
     [ 0.2395362   0.30088416 -0.35621717  0.06852048  0.1083728  -0.09372214
       0.25995959 -0.39613174 -0.39902922  0.26782704]
     [-0.31436238 -0.2981721   0.12518176 -0.30121827 -0.06144361  0.3848255
       0.46794533  0.23381171  0.14337317 -0.37994111]]


#### 59. 如何按第n列对数组进行排序？ (★★☆)


```python
# Author: Steve Tjoa

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
```

    [[7 9 1]
     [4 9 9]
     [1 7 6]]
    [[1 7 6]
     [7 9 1]
     [4 9 9]]


#### 60. 如何判断给定的2D数组是否具有空列？(★★☆)


```python
# Author: Warren Weckesser

Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())
```

    False


#### 61. 从数组中的给定值中查找最接近的值 (★★☆)


```python
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
```

    0.5753223794637276


#### 62. 考虑两个形状为（1,3）和（3,1）的数组，如何使用迭代器计算它们的总和？ (★★☆)


```python
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])
```

    [[0 1 2]
     [1 2 3]
     [2 3 4]]


#### 63. 创建一个具有name属性的数组类 (★★☆)


```python
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
```

    range_10


#### 64. 考虑一个给定的向量，如何为每个由第二个向量索引的元素添加1（小心重复索引）？ (★★★)


```python
# Author: Brett Olsen

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

# Another solution
# Author: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)
```

    [4. 1. 2. 5. 3. 2. 4. 3. 4. 2.]
    [7. 1. 3. 9. 5. 3. 7. 5. 7. 3.]


#### 65. 如何基于索引列表（I）将向量（X）的元素累积到数组（F）？ (★★★)


```python
# Author: Alan G Isaac

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
```

    [0. 7. 0. 6. 5. 0. 0. 0. 0. 3.]


#### 66. 考虑（dtype = ubyte）的（w，h，3）图像，计算唯一颜色的数量 (★★★)


```python
# Author: Nadav Horesh

w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
#Note that we should compute 256*256 first.
#Otherwise numpy will only promote F.dtype to 'uint16' and overfolw will occur
F = I[...,0]*(256*256) + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(n)
```

    8


#### 67. 考虑四维数组，如何一次得到最后两个轴的总和？ (★★★)


```python
A = np.random.randint(0,10,(3,4,3,4))
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = A.sum(axis=(-2,-1))
print(sum)
# solution by flattening the last two dimensions into one
# (useful for functions that don't accept tuples for axis argument)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```

    [[72 80 48 56]
     [51 63 65 54]
     [46 67 54 43]]
    [[72 80 48 56]
     [51 63 65 54]
     [46 67 54 43]]


#### 68. 考虑一维向量D，如何使用描述子集索引的相同大小的向量S来计算D的子集的均值？ (★★★)


```python
# Author: Jaime Fernández del Río

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# Pandas solution as a reference due to more intuitive code
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```

    [0.56424423 0.49555383 0.46335751 0.56006403 0.60439609 0.60102577
     0.56262121 0.53673865 0.43915693 0.41721056]
    0    0.564244
    1    0.495554
    2    0.463358
    3    0.560064
    4    0.604396
    5    0.601026
    6    0.562621
    7    0.536739
    8    0.439157
    9    0.417211
    dtype: float64


#### 69. 如何获得点积的对角线？ (★★★)


```python
# Author: Mathieu Blondel

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

# Slow version
np.diag(np.dot(A, B))

# Fast version
np.sum(A * B.T, axis=1)

# Faster version
np.einsum("ij,ji->i", A, B)
```




    array([1.26769213, 1.83285132, 1.01691636, 1.17953308, 1.17095184])



#### 70. 考虑向量 \[1,2,3,4,5]，如何构建一个新的向量，在每个值之间插入3个连续的零？ (★★★)


```python
# Author: Warren Weckesser

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```

    [1. 0. 0. 0. 2. 0. 0. 0. 3. 0. 0. 0. 4. 0. 0. 0. 5.]


#### 71. 考虑一个维度数组（5,5,3），如何将它乘以一个维数为（5,5）的数组？(★★★)


```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```

    [[[2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]]

     [[2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]]

     [[2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]]

     [[2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]]

     [[2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]
      [2. 2. 2.]]]


#### 72. 如何交换数组的两行？ (★★★)


```python
# Author: Eelco Hoogendoorn

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```

    [[ 5  6  7  8  9]
     [ 0  1  2  3  4]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]


#### 73. 考虑一组描述10个三角形（具有共享顶点）的10个三元组，找到组成所有三角形的唯一线段的集合 (★★★)


```python
# Author: Nicolas P. Rougier

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```

    [( 0, 70) ( 0, 91) ( 3, 29) ( 3, 53) ( 3, 90) ( 3, 98) (24, 55) (24, 64)
     (27, 63) (27, 85) (29, 35) (29, 40) (29, 90) (35, 40) (35, 52) (35, 74)
     (38, 42) (38, 81) (42, 81) (52, 74) (53, 56) (53, 62) (53, 67) (53, 92)
     (53, 98) (55, 64) (56, 67) (62, 92) (63, 85) (70, 91)]


#### 74. 给定一个bincount的数组C，如何生成一个数组A，使得 np.bincount(A) == C? (★★★)


```python
# Author: Jaime Fernández del Río

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```

    [1 1 2 3 4 4 6]


#### 75. 如何使用数组上的滑动窗口计算平均值？ (★★★)


```python
# Author: Jaime Fernández del Río

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
```

    [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18.]


#### 76. 考虑一维数组Z，构建一个二维数组，其第一行为(Z\[0\],Z\[1\],Z\[2\])，每个后续行移1 行 (最后一行应该是Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)


```python
# Author: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)
```

    [[0 1 2]
     [1 2 3]
     [2 3 4]
     [3 4 5]
     [4 5 6]
     [5 6 7]
     [6 7 8]
     [7 8 9]]


#### 77. 如何否定布尔值，或改变浮点数的符号？ (★★★)


```python
# Author: Nathaniel J. Smith

Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
```




    array([ 5.77110369e-01,  8.54166972e-01, -4.60133644e-01,  2.41535767e-01,
            6.78304096e-01, -1.24540357e-01, -4.31974561e-04,  9.35034232e-01,
           -7.67291343e-01,  3.49412740e-01,  5.29319222e-02,  5.01403521e-01,
           -1.83014069e-01,  2.84273592e-01,  4.90344783e-01,  9.35893921e-01,
            9.25496766e-01,  8.36447456e-01, -8.74583470e-01, -8.70245868e-01,
           -1.33664699e-01,  9.51915734e-01,  6.29171977e-01, -2.01972265e-01,
           -1.48297051e-01, -5.40488235e-01, -9.13252507e-01, -6.59781707e-01,
            6.49629587e-01, -7.08994926e-01,  7.65498684e-01, -3.69025121e-01,
           -1.09482919e-01, -4.40156883e-01,  3.54815222e-01,  1.06050733e-02,
            1.08040405e-01, -6.45636526e-01,  7.42230089e-01,  7.05363666e-01,
            6.19865270e-01,  2.28530862e-01, -4.93649797e-01,  2.18275863e-01,
            7.17599345e-01, -5.69758251e-01, -2.65298780e-01,  7.89558470e-02,
           -3.91324145e-01,  2.95500309e-01,  2.69348734e-01, -2.38219086e-01,
            5.62370170e-01, -5.71723588e-01, -2.06076844e-01,  9.20130949e-01,
            8.00588812e-01, -3.70254631e-01, -1.55522928e-01, -2.35975970e-01,
           -3.88199712e-01,  5.31439178e-01, -6.94972149e-01, -1.50367379e-01,
           -7.36836585e-01,  5.58918700e-01,  6.49538986e-01, -4.61944630e-02,
           -5.22596309e-01, -3.24928811e-01,  2.91657876e-02, -1.86706303e-01,
            2.75972849e-02, -9.69889798e-01, -3.86160457e-01, -2.99781660e-01,
           -9.22491810e-01,  2.93775654e-01,  9.86920527e-01,  5.22669798e-01,
            1.27741485e-01, -3.99541301e-01,  7.68256625e-01, -5.62152698e-02,
            1.48450944e-01, -9.80402287e-02,  1.38121829e-02,  3.87544569e-01,
           -2.92381246e-01,  3.92330830e-01, -4.40261254e-01,  3.01719596e-01,
            4.65732247e-01,  3.89391993e-01,  3.46932543e-01, -1.34033739e-01,
           -8.63330386e-01, -5.70192069e-01,  7.11072357e-01,  7.29465325e-01])



#### 78. 考虑2组点P0，P1描述线（2d）和点p，如何计算从p到每条线的距离i (P0\[i\],P1\[i\])? (★★★)


```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
```

    [ 6.92907313  4.5753023   6.4993971   0.04557899 11.55250429 10.3699155
      6.75355305  2.86028759 13.51095661 12.4618502 ]


#### 79. 考虑2组点P0，P1描述线（2d）和一组点P，如何计算从每个点j (P\[j\])到每条线的距离i (P0\[i\],P1\[i\])? (★★★)


```python
# Author: Italmassov Kuanysh

# based on distance function from previous question
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
```

    [[ 0.89809952  5.63125986  3.51163075  2.00882249 14.66193707  5.35014287
      14.21303109  9.24435594  1.47293346  8.9194382 ]
     [ 9.66508933  1.83737105  4.70842644  4.983849    5.96151774  1.25203828
       5.38980459  3.57783403  7.1475612  10.34846067]
     [ 0.37806534  7.1849528   0.95693607  3.74699372 15.31624475  3.20339424
      14.46968481  7.44232539  1.15788903  6.36868234]
     [14.58064627  3.53052684  0.19861055  5.93241914  1.39371495  6.41297999
       0.15629279  4.40133077 13.83693904  5.61141373]
     [ 7.47676762  2.56120118 10.19717105  6.3221376   7.81114012  7.84371237
       8.21366026  9.97453117  3.0701716  15.7379628 ]
     [ 4.48318396  1.97787547  5.34105988  1.56469099 11.02950074  4.96368524
      10.74885023  8.08073551  1.60638618 10.83427471]
     [18.43835598 10.04619104  7.5493422  12.85710216  2.83658111  1.26784118
       3.26327474  0.67810421 15.22747405 13.41004964]
     [ 0.12078809 11.62774451  8.09585905  8.9321756  16.3113236   5.2085546
      14.00690316  0.03007727  1.34102085  2.63324757]
     [ 1.90122096  4.70295559  3.81347001  1.12085503 13.6572556   5.03990602
      13.22118324  8.73795019  0.54144081  9.24667283]
     [11.12963793  2.27604364  3.09929837  5.18807903  4.6087669   1.17324693
       3.72159332  1.07385957  9.18955444  8.79102895]]


#### 80. 考虑一个任意数组，编写一个函数，提取具有固定形状的子部分并以给定元素为中心（必要时使用“fill”值填充） (★★★)


```python
# Author: Nicolas Rougier

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```

    [[8 3 9 2 0 9 1 1 1 4]
     [9 3 2 2 1 3 8 7 1 9]
     [1 5 6 6 3 5 0 9 5 5]
     [3 6 4 9 7 8 6 1 6 5]
     [7 9 3 9 9 1 8 9 6 9]
     [9 0 3 0 7 4 9 7 2 9]
     [8 1 9 9 8 9 2 9 9 0]
     [2 5 4 4 7 1 0 0 1 1]
     [9 9 0 4 8 2 3 4 1 1]
     [3 5 6 4 5 2 7 5 1 0]]
    [[0 0 0 0 0]
     [0 8 3 9 2]
     [0 9 3 2 2]
     [0 1 5 6 6]
     [0 3 6 4 9]]


#### 81. 考虑一个数组 Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], 如何生成一个数组 R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]?


```python
# Author: Stefan van der Walt

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)
```

    [[ 1  2  3  4]
     [ 2  3  4  5]
     [ 3  4  5  6]
     [ 4  5  6  7]
     [ 5  6  7  8]
     [ 6  7  8  9]
     [ 7  8  9 10]
     [ 8  9 10 11]
     [ 9 10 11 12]
     [10 11 12 13]
     [11 12 13 14]]


#### 82. 计算矩阵排名(★★★)


```python
# Author: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)
```

    10


#### 83. 如何在数组中找到最常见的值？


```python
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
```

    8


#### 84. 从随机10x10矩阵中提取所有连续的3x3块(★★★)


```python
# Author: Chris Barker

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)
```

    [[[[2 2 2]
       [0 4 0]
       [3 4 3]]

      [[2 2 0]
       [4 0 2]
       [4 3 4]]

      [[2 0 4]
       [0 2 2]
       [3 4 0]]

      [[0 4 1]
       [2 2 4]
       [4 0 1]]

      [[4 1 2]
       [2 4 0]
       [0 1 2]]

      [[1 2 3]
       [4 0 1]
       [1 2 1]]

      [[2 3 3]
       [0 1 4]
       [2 1 1]]

      [[3 3 2]
       [1 4 1]
       [1 1 4]]]


     [[[0 4 0]
       [3 4 3]
       [3 4 0]]

      [[4 0 2]
       [4 3 4]
       [4 0 0]]

      [[0 2 2]
       [3 4 0]
       [0 0 3]]

      [[2 2 4]
       [4 0 1]
       [0 3 0]]

      [[2 4 0]
       [0 1 2]
       [3 0 3]]

      [[4 0 1]
       [1 2 1]
       [0 3 4]]

      [[0 1 4]
       [2 1 1]
       [3 4 3]]

      [[1 4 1]
       [1 1 4]
       [4 3 0]]]


     [[[3 4 3]
       [3 4 0]
       [2 4 1]]

      [[4 3 4]
       [4 0 0]
       [4 1 2]]

      [[3 4 0]
       [0 0 3]
       [1 2 0]]

      [[4 0 1]
       [0 3 0]
       [2 0 2]]

      [[0 1 2]
       [3 0 3]
       [0 2 0]]

      [[1 2 1]
       [0 3 4]
       [2 0 4]]

      [[2 1 1]
       [3 4 3]
       [0 4 3]]

      [[1 1 4]
       [4 3 0]
       [4 3 2]]]


     [[[3 4 0]
       [2 4 1]
       [4 0 2]]

      [[4 0 0]
       [4 1 2]
       [0 2 0]]

      [[0 0 3]
       [1 2 0]
       [2 0 2]]

      [[0 3 0]
       [2 0 2]
       [0 2 4]]

      [[3 0 3]
       [0 2 0]
       [2 4 4]]

      [[0 3 4]
       [2 0 4]
       [4 4 0]]

      [[3 4 3]
       [0 4 3]
       [4 0 0]]

      [[4 3 0]
       [4 3 2]
       [0 0 4]]]


     [[[2 4 1]
       [4 0 2]
       [2 2 3]]

      [[4 1 2]
       [0 2 0]
       [2 3 4]]

      [[1 2 0]
       [2 0 2]
       [3 4 0]]

      [[2 0 2]
       [0 2 4]
       [4 0 2]]

      [[0 2 0]
       [2 4 4]
       [0 2 2]]

      [[2 0 4]
       [4 4 0]
       [2 2 4]]

      [[0 4 3]
       [4 0 0]
       [2 4 4]]

      [[4 3 2]
       [0 0 4]
       [4 4 0]]]


     [[[4 0 2]
       [2 2 3]
       [4 4 2]]

      [[0 2 0]
       [2 3 4]
       [4 2 2]]

      [[2 0 2]
       [3 4 0]
       [2 2 1]]

      [[0 2 4]
       [4 0 2]
       [2 1 1]]

      [[2 4 4]
       [0 2 2]
       [1 1 3]]

      [[4 4 0]
       [2 2 4]
       [1 3 4]]

      [[4 0 0]
       [2 4 4]
       [3 4 3]]

      [[0 0 4]
       [4 4 0]
       [4 3 1]]]


     [[[2 2 3]
       [4 4 2]
       [0 4 2]]

      [[2 3 4]
       [4 2 2]
       [4 2 1]]

      [[3 4 0]
       [2 2 1]
       [2 1 0]]

      [[4 0 2]
       [2 1 1]
       [1 0 2]]

      [[0 2 2]
       [1 1 3]
       [0 2 4]]

      [[2 2 4]
       [1 3 4]
       [2 4 1]]

      [[2 4 4]
       [3 4 3]
       [4 1 0]]

      [[4 4 0]
       [4 3 1]
       [1 0 1]]]


     [[[4 4 2]
       [0 4 2]
       [0 2 2]]

      [[4 2 2]
       [4 2 1]
       [2 2 3]]

      [[2 2 1]
       [2 1 0]
       [2 3 3]]

      [[2 1 1]
       [1 0 2]
       [3 3 0]]

      [[1 1 3]
       [0 2 4]
       [3 0 0]]

      [[1 3 4]
       [2 4 1]
       [0 0 2]]

      [[3 4 3]
       [4 1 0]
       [0 2 1]]

      [[4 3 1]
       [1 0 1]
       [2 1 3]]]]


#### 85. 创建一个二维数组子类，使 Z\[i,j\] == Z\[j,i\]


```python
# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```

    [[ 3 10  4  5  5]
     [10  1  8 13  9]
     [ 4  8  6 42 16]
     [ 5 13 42  5  3]
     [ 5  9 16  3  9]]


#### 86. 考虑一组具有形状（n，n）的p矩阵和一组具有形状（n，1）的p向量。 如何一次计算p矩阵乘积的总和？ （结果有形状（n，1）） (★★★)


```python
# Author: Stefan van der Walt

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
```

    [[200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]
     [200.]]


#### 87. 考虑一个16x16阵列，如何获得块总和（块大小为4x4）？ (★★★)


```python
# Author: Robert Kern

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
```

    [[16. 16. 16. 16.]
     [16. 16. 16. 16.]
     [16. 16. 16. 16.]
     [16. 16. 16. 16.]]


#### 88. 如何使用numpy数组实现游戏生命？ (★★★)


```python
# Author: Nicolas Rougier

def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)
```

    [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0
      0 0 0 0 0 0 0 1 1 1 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1 1 0 1 1 1 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 1 1 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0
      0 0 0 1 1 0 0 0 0 0 0 0 0 0]
     [0 0 1 1 0 0 0 0 1 0 0 0 0 1 1 0 1 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 1 0 1 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 1 1 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 1 0 0 1 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 1 0 0 1 0]
     [0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 1 1 0 0]
     [0 0 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 0 1 1 1 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 1 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1
      0 0 0 0 0 0 0 0 0 1 1 0 0 0]
     [0 0 0 0 1 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1
      0 0 0 0 0 0 0 0 0 1 1 0 0 0]
     [0 0 0 0 1 0 1 0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 1 0 1 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 1 0 1 1 0 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      1 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 1 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 1 0 0 0 1 0 0 1 1 0 0 0 0]
     [0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 1 1 1 0 0 1 1 1 0 0 0]
     [0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 1 1 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 1 1 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 1 0 0 1 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 1 0 1 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 1 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0]]


#### 89. 如何获取数组的n个最大值 (★★★)


```python
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Slow
print (Z[np.argsort(Z)[-n:]])

# Fast
print (Z[np.argpartition(-Z,n)[:n]])
```

    [9995 9996 9997 9998 9999]
    [9997 9999 9998 9996 9995]


#### 90. 给定任意数量的向量，构建笛卡尔积（每个项的每个组合） (★★★)


```python
# Author: Stefan Van der Walt

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
```

    [[1 4 6]
     [1 4 7]
     [1 5 6]
     [1 5 7]
     [2 4 6]
     [2 4 7]
     [2 5 6]
     [2 5 7]
     [3 4 6]
     [3 4 7]
     [3 5 6]
     [3 5 7]]


#### 91. 如何从常规数组创建记录数组？ (★★★)


```python
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
```

    [(b'Hello', 2.5, 3) (b'World', 3.6, 2)]


#### 92. 考虑一个大的向量Z，使用3种不同的方法将Z计算为3的幂 (★★★)


```python
# Author: Ryan G.

x = np.random.rand(int(5e7))

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)
```

    1.26 s ± 38.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    171 ms ± 31.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    288 ms ± 27.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


#### 93. 考虑形状 (8,3) 和 (2,2) 的两个阵列A和B. 如何查找包含B的每一行元素的A行，而不管B中元素的顺序如何？ (★★★)


```python
# Author: Gabe Schwartz

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```

    [0 5 6]


#### 94. 考虑10x3矩阵，提取具有不等值的行 (e.g. \[2,2,3\]) (★★★)


```python
# Author: Robert Kern

Z = np.random.randint(0,5,(10,3))
print(Z)
# solution for arrays of all dtypes (including string arrays and record arrays)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
# soluiton for numerical arrays only, will work for any number of columns in Z
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)
```

    [[1 0 1]
     [1 3 3]
     [2 0 3]
     [0 0 4]
     [4 1 1]
     [2 3 4]
     [0 0 1]
     [1 0 3]
     [2 1 1]
     [3 1 2]]
    [[1 0 1]
     [1 3 3]
     [2 0 3]
     [0 0 4]
     [4 1 1]
     [2 3 4]
     [0 0 1]
     [1 0 3]
     [2 1 1]
     [3 1 2]]
    [[1 0 1]
     [1 3 3]
     [2 0 3]
     [0 0 4]
     [4 1 1]
     [2 3 4]
     [0 0 1]
     [1 0 3]
     [2 1 1]
     [3 1 2]]


#### 95. 将int的向量转换为矩阵二进制表示 (★★★)


```python
# Author: Warren Weckesser

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Author: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```

    [[0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 1 0]
     [0 0 0 0 0 0 1 1]
     [0 0 0 0 1 1 1 1]
     [0 0 0 1 0 0 0 0]
     [0 0 1 0 0 0 0 0]
     [0 1 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0]]
    [[0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 1 0]
     [0 0 0 0 0 0 1 1]
     [0 0 0 0 1 1 1 1]
     [0 0 0 1 0 0 0 0]
     [0 0 1 0 0 0 0 0]
     [0 1 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0]]


#### 96. 给定一个二维数组，如何提取唯一的行？(★★★)


```python
# Author: Jaime Fernández del Río

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)
```

    [[0 0 1]
     [0 1 1]
     [1 0 0]
     [1 1 1]]


#### 97. 考虑2个向量A & B，写出einsum等效的inner，outer，sum和mul函数(★★★)


```python
# Author: Alex Riley
# Make sure to read: http://ajcr.net/Basic-guide-to-einsum/

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```




    array([[0.03531573, 0.07042564, 0.11028219, 0.02372113, 0.15768101,
            0.05898971, 0.08460937, 0.09321772, 0.1484096 , 0.13047303],
           [0.05352301, 0.10673411, 0.16713901, 0.03595074, 0.23897463,
            0.08940229, 0.12823037, 0.14127681, 0.22492328, 0.19773938],
           [0.04210276, 0.08396015, 0.13147639, 0.02827989, 0.18798438,
            0.07032644, 0.10086973, 0.11113244, 0.17693118, 0.15554754],
           [0.05967586, 0.11900395, 0.18635281, 0.04008353, 0.26644645,
            0.09967972, 0.14297135, 0.15751758, 0.25077979, 0.22047092],
           [0.01795077, 0.03579693, 0.05605577, 0.01205731, 0.0801483 ,
            0.02998411, 0.04300643, 0.047382  , 0.0754357 , 0.06631865],
           [0.16279527, 0.32464182, 0.50836899, 0.10934755, 0.7268638 ,
            0.27192548, 0.3900247 , 0.4297067 , 0.68412529, 0.60144292],
           [0.10046066, 0.20033586, 0.31371356, 0.06747817, 0.4485463 ,
            0.1678047 , 0.24068352, 0.2651712 , 0.42217244, 0.3711493 ],
           [0.01904045, 0.03796994, 0.05945857, 0.01278923, 0.08501361,
            0.03180426, 0.04561709, 0.05025827, 0.08001494, 0.07034445],
           [0.0040046 , 0.00798587, 0.01250537, 0.00268984, 0.01788013,
            0.0066891 , 0.00959422, 0.01057036, 0.0168288 , 0.0147949 ],
           [0.11924398, 0.23779305, 0.37236919, 0.08009469, 0.53241188,
            0.19917948, 0.28568459, 0.31475078, 0.50110685, 0.44054381]])



#### 98. 考虑两个矢量(X,Y)描述的路径，如何使用等距样本对其进行采样 (★★★)?


```python
# Author: Bas Swinckels

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)
```

#### 99. 给定整数n和2D数组X，从X中选择可以解释为具有n度的多项分布的绘制的行，即，仅包含整数并且总和为n的行。 (★★★)


```python
# Author: Evgeni Burovski

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```

    [[2. 0. 1. 1.]]


#### 100. 计算一维阵列X的平均值的自举95％置信区间（即，重新采样具有替换N次的阵列的元素，计算每个样本的平均值，然后计算均值上的百分位数）。(★★★)


```python
# Author: Jessica B. Hamrick

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```

    [-0.30412351  0.12394569]

