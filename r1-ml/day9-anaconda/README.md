# 2019年最新最全的 Anaconda 安装指南
---

> 君子生非异也，善假于物也
>                                                            -- 荀子

从 `Anaconda` 官文网站 <https://www.anaconda.com/download> 下载操作系统对就的安装文件，选择 Python 3.7 版本。

`Anaconda` 默认安装了许多有用的数据科学工具和 `Python` 库。可以直接测试 `sklearn`。

# Windows

下载地址: <https://www.anaconda.com/download/#windows>

下载后缀为 `exe` 格式的文件，下载完成后双击开始安装，一直按 *Next* 即可。

![1.PNG](https://upload-images.jianshu.io/upload_images/910914-5341a1aff83e4654.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![2.PNG](https://upload-images.jianshu.io/upload_images/910914-3aacb364ae69709d.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![3.PNG](https://upload-images.jianshu.io/upload_images/910914-7bc638710cf94e59.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![4.PNG](https://upload-images.jianshu.io/upload_images/910914-4cbfd3f89b6d5c29.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![5.PNG](https://upload-images.jianshu.io/upload_images/910914-6d21f44aa8b85980.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![6.PNG](https://upload-images.jianshu.io/upload_images/910914-9b7ce0c6b20f48db.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![7.PNG](https://upload-images.jianshu.io/upload_images/910914-fd75bb4747f78225.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![8.PNG](https://upload-images.jianshu.io/upload_images/910914-b598ad810bc67b34.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![9.PNG](https://upload-images.jianshu.io/upload_images/910914-509cc16d06a168cf.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从开始菜单启动 `Anaconda Navigator`

![10.PNG](https://upload-images.jianshu.io/upload_images/910914-9f08d526667845bd.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

新建 `Python 3` Notebook

![11.PNG](https://upload-images.jianshu.io/upload_images/910914-656b65bd943e91c7.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

查看 sklearn 版本

![sklearn_version.PNG](https://upload-images.jianshu.io/upload_images/910914-c87b2db158a01f0c.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
import sklearn
sklearn.show_versions()
Could not locate executable g77
Could not locate executable f77
Could not locate executable ifort
Could not locate executable ifl
Could not locate executable f90
Could not locate executable DF
Could not locate executable efl
Could not locate executable gfortran
Could not locate executable f95
Could not locate executable g95
Could not locate executable efort
Could not locate executable efc
Could not locate executable flang
don't know how to compile Fortran code on platform 'nt'

System:
    python: 3.7.1 (default, Dec 10 2018, 22:54:23) [MSC v.1915 64 bit (AMD64)]
executable: D:\Anaconda3\python.exe
   machine: Windows-10-10.0.17134-SP0

BLAS:
    macros:
  lib_dirs:
cblas_libs: cblas

Python deps:
       pip: 18.1
setuptools: 40.6.3
   sklearn: 0.20.1
     numpy: 1.15.4
     scipy: 1.1.0
    Cython: 0.29.2
    pandas: 0.23.4
```

# macOS

下载地址: <https://www.anaconda.com/download/#macos>

下载后缀为 `pkg` 格式的文件，下载完成后双击开始安装。

1. Anaconda 安装器提示

![1.png](https://upload-images.jianshu.io/upload_images/910914-2eae6f16d7d0194d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2.  介绍 Anaconda 安装器

![2.png](https://upload-images.jianshu.io/upload_images/910914-02a28c24ccc44d0d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3.  请先阅读安装 Anaconda 相关信息

![3.png](https://upload-images.jianshu.io/upload_images/910914-3a737e556151bf56.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

4.  查看许可协议

![4.png](https://upload-images.jianshu.io/upload_images/910914-94d0d321a8c28e21.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

5.  同意条款

![5.png](https://upload-images.jianshu.io/upload_images/910914-b4bbb4e23fb39419.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

6.  安装位置

![6.png](https://upload-images.jianshu.io/upload_images/910914-c6a7a0ff24551229.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

7.  正式安装 Anaconda

![7.png](https://upload-images.jianshu.io/upload_images/910914-2a83856c071a18d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

8.  暂时不安装 VSCode，如何需要，稍后可以通过 Anaconda-Navigator 安装

![8.png](https://upload-images.jianshu.io/upload_images/910914-72d4f24f0708ba0c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

9.  安装成功

![9.png](https://upload-images.jianshu.io/upload_images/910914-e59444df87bb34ae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

10. `DashBoard` 里面搜索 `Anaconda-Navigator`

![10.png](https://upload-images.jianshu.io/upload_images/910914-04429ea378b86371.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

11. 显示主界面，这里可以安装 `VSCode` 等应用程序

![11.png](https://upload-images.jianshu.io/upload_images/910914-a7fe989d536fb458.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

12. 查看 `sklearn` 版本

![sklearn.png](https://upload-images.jianshu.io/upload_images/910914-94a94ab8b3ebaabe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
import sklearn

sklearn.show_versions()

System:
    python: 3.7.1 (default, Dec 14 2018, 13:28:58)  [Clang 4.0.1 (tags/RELEASE_401/final)]
executable: /Users/iosdevlog/anaconda3/bin/python
   machine: Darwin-18.2.0-x86_64-i386-64bit

BLAS:
    macros: SCIPY_MKL_H=None, HAVE_CBLAS=None
  lib_dirs: /Users/iosdevlog/anaconda3/lib
cblas_libs: mkl_rt, pthread

Python deps:
       pip: 18.1
setuptools: 40.6.3
   sklearn: 0.20.1
     numpy: 1.15.4
     scipy: 1.1.0
    Cython: 0.29.2
    pandas: 0.23.4
```

# Linux

下载地址：<https://www.anaconda.com/download/#linux>

下载后缀为 `sh` 格式的文件，下载完成后打开终端进入下载目标，输入以下命令开始安装。

```
$ wget -c https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
$ chmod +x Anaconda3-2018.12-Linux-x86_64.sh
$ ./Anaconda3-2018.12-Linux-x86_64.sh
```

1. 运行安装程序

![1.png](https://upload-images.jianshu.io/upload_images/910914-56da773397de8c56.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2. 空格翻页查看 License

![2.png](https://upload-images.jianshu.io/upload_images/910914-d44e5cd33762dccd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3. 输入 *yes* 接受 License

![3.png](https://upload-images.jianshu.io/upload_images/910914-e85b113d8c784d7e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

4. 安装位置，这里使用默认

![4.png](https://upload-images.jianshu.io/upload_images/910914-745455393cae02af.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

5. 配置 Anaconda 环境到 `~/.bashrc`

![5.png](https://upload-images.jianshu.io/upload_images/910914-d9c970b55acc1c81.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

6. 先不安装 VSCode

![6.png](https://upload-images.jianshu.io/upload_images/910914-c3dbb7454e596ff7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

7. 重新加载 `~/.bashrc`，运行 anaconda-navigator

![7.png](https://upload-images.jianshu.io/upload_images/910914-c51cd13cf63f167e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

8. 显示主界面，这里可以安装 `VSCode` 等应用程序

![8.png](https://upload-images.jianshu.io/upload_images/910914-70844cca6221a92f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

9. 查看 `sklearn` 版本

![9.png](https://upload-images.jianshu.io/upload_images/910914-248ff8df33d32d44.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
$ python
Python 3.7.1 (default, Dec 14 2018, 19:28:38)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sklearn
>>> sklearn.show_versions()

System:
    python: 3.7.1 (default, Dec 14 2018, 19:28:38)  [GCC 7.3.0]
executable: /home/iosdevlog/anaconda3/bin/python
   machine: Linux-4.13.0-45-generic-x86_64-with-debian-buster-sid

BLAS:
    macros: SCIPY_MKL_H=None, HAVE_CBLAS=None
  lib_dirs: /home/iosdevlog/anaconda3/lib
cblas_libs: mkl_rt, pthread

Python deps:
       pip: 18.1
setuptools: 40.6.3
   sklearn: 0.20.1
     numpy: 1.15.4
     scipy: 1.1.0
    Cython: 0.29.2
    pandas: 0.23.4
>>>
```
