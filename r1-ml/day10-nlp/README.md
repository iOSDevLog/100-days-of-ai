---
layout: post
title: 自然语言处理简明教程
author: iosdevlog
date: 2019-01-03 10:46:55 +0800
description: ""
category: 机器学习
tags: []
---

# 自然语言处理简介

现在，让我们先从介绍自然语言处理(NLP)开始吧。众所周知，语言是人们日常生 活的核心部分，任何与语言问题相关的工作都会显得非常有意思。希望这本书能带你领略 到 NLP 的风采，并引起学习 NLP 的兴趣。首先，我们需要来了解一下该领域中的一些令 人惊叹的概念，并在工作中实际尝试一些具有挑战性的 NLP 应用。
在英语环境中，语言处理研究这一领域通常被简称为 NLP。对语言有深入研究的人通 常被叫作语言学家，而“计算机语言学家”这个专用名词则指的是将计算机科学应用于语 言处理领域的人。因此从本质上来说，一个计算机语言学家应该既有足够的语言理解能力， 同时还可以用其计算机技能来模拟出语言的不同方面。虽然计算机语言学家主要研究的是 语言处理理论，但 NLP 无疑是对计算机语言学的具体应用。
NLP 多数情况下指的是计算机上各种大同小异的语言处理应用，以及用 NLP 技术所构 建的实际应用程序。在实践中，NLP 与教孩子学语言的过程非常类似。其大多数任务(如 对单词、语句的理解，形成语法和结构都正确的语句等)对于人类而言都是非常自然的能 力。但对于 NLP 来说，其中有一些任务就必须要转向标识化处理、语块分解、词性标注、 语法解析、机器翻译及语音识别等这些领域的一部分，且这些任务有一大部分还仍是当前 计算机领域中非常棘手的挑战。在本书中，我们将更侧重于讨论 NLP 的实用方面，因此我 们会假设读者在 NLP 上已经有了一些背景知识。所以，读者最好在最低限度上对编程语言 有一点了解，并对 NLP 和语言学有一定的兴趣。

目前，NLP 已被认为是业界最为稀缺的技能之一。自大数据 的概念问世之后，我们所面对的主要挑战是——业界需要越来越多不仅能处理结构化数据， 同时也能处理半结构化或非结构化数据的人才。对于我们所生产出来的那些博客、微博、 Facebook 订阅、聊天信息、E-mail 以及网络评论等，各公司都在致力于收集所有不同种类 的数据，以便建立更好的客户针对性，形成有意义的见解。而要想处理所有的这些非结构 化数据源，我们就需要掌握一些 NLP 技能的人员。
身处信息时代，我们甚至不能想象生活中没有 Google 会是什么样子。我们会因一些最基本的事情而用到 Siri;我们会需要用垃圾过滤器来过滤垃圾邮件;我们会需要在自己的 Word 文档中用到拼写检查器等。在现实世界中所要用到的 NLP 应用数不胜数。

在这里，我们可以再列举一些令人惊叹的 NLP 应用实例。虽然你很可能已经用过它们，
但未必知道这些应用是基于 NLP 技术的。

• 拼写校正(MS Word/其他编辑器)

• 搜索引擎(Google、Bing、Yahoo!、WolframAlpha)

• 语音引擎(Siri、Google Voice)

• 垃圾邮件分类(所有电子邮件服务)

• 新闻订阅(Google、Yahoo!等)

• 机器翻译(Google 翻译与其他类似服务)

• IBM Watson1


构建上述这些应用都需要非常具体的技能，需要优秀的语言理解能力和能有效处理这 些语言的工具。因此，这些不仅是各 NLP 最具优势领域的未来趋势，同时也是我们用 NLP 这种最独特技能所能创建的应用种类。

# Natural Language Tool Kit (NLTK)

NLTK 库是一个非常易学的工具包，这得益于 Python 本身 非常平缓的学习曲线(毕竟 NLTK 是用它编写的)，人们学习起来会非常快。NLTK 库中收纳了 NLP 领域中的绝大部分任务，它们都被实现得非常优雅，且易于使用。正是出于上述
的这些原因，NLTK 如今已成为了 NLP 社区最流行的库之一。

# 正则表达式

对 NLP 爱好者来说，正则表达式是另一个非常重要的技能。正则表达式(regular expression) 是一种能对字符串进行有效匹配的模式。我们会大量使用这种模式，以求从大量凌乱的文 本数据中提取出有意义的信息。下面，我们就来整体浏览一下你将会用到哪些正则表达式。 其实，我这一生至今所用过的正则表达式无非也就是以下这些。

• (句点):该表达式用于匹配除换行符\n 外的任意单字符

• \w:该表达式用于匹配某一字符或数字，相当于[a-z A-Z 0-9]

• \W(大写 W):该表达式用于匹配任意非单词性字符

• \s(小写 s):用于匹配任意单个空白字符，包括换行、返回、制表等，相当于[\n\r\t\f]

• \S:该表达式用于匹配单个任意非空白字符

• \t:该表达式用于匹配制表符

• \n:该表达式用于匹配换行符

• \r:该表达用于匹配返回符

• \d:该表达式用于匹配十进制数字，即[0-9]

• ^:该表达式用于匹配相关字符串的开始位置

• $:该表达式用于匹配相关字符串的结尾位置

• \:该表达式用来抵消特殊字符的特殊性。如要匹配$符号，就在它前面加上\

# 文本清理

一旦我们将各种数据源解析成了文本形式，接下来所要面临的挑战就是要使这些原生 数据体现出它们的意义。文本清理就泛指针对文本所做的绝大部分清理、与相关数据源的 依赖关系、性能的解析和外部噪声等。

## 语句分离

* 字
* 词
* 句
* 段
* 篇
* 章

语句分离是将大段的语句分成句子。

* 段/篇/章 -> 句

## 标识化处理

可以理解为编译原理里面的**词法分析**，把语句分成**标记(token)**。

* 句 -> 词

标识器(tokenizer)

1. 第一种是 word_tokenize()，这是我们的默 认选择，基本上能应付绝大多数的情况。
2. 另一选择是 regex_tokenize()，这是一个为用户特 定需求设计的、自定义程度更高的标识器。

## 词干提取（词根化）

所谓词干提取(stemming)，顾名思义就是一个修剪枝叶的过程。这是很有效的方法， 通过运用一些基本规则，我们可以在修剪枝叶的过程中得到所有的分词。词干提取是一种 较为粗糙的规则处理过程，我们希望用它来取得相关分词的各种变化。

例如 eat 这个单词 就会有像 eating、eaten、eats 等变化。在某些应用中，我们是没有必要区分 eat 和 eaten 之 间的区别的，所以通常会用词干提取的方式将这种语法上的变化归结为相同的词根。

由此 可以看出，我们之所以会用词干提取方法，就是因为它的简单，而对于更复杂的语言案 例或更复杂的 NLP 任务，我们就必须要改用词形还原(lemmatization)的方法了。词形 还原是一种更为健全、也更有条理的方法，以便用于应对相关词根的各种语法上的变化。

一个拥有基本规则的词干提取器，在像移除-s/es、-ing 或-ed 这类事情上都可以达到 70%以 上的精确度，而 Porter 词干提取器使用了更多的规则，自然在执行上会得到很不错的精确度。

## 停用词移除

停用词移除(Stop word removal)是在不同的 NLP 应用中最常会用到的预处理步骤之 一。

该步骤的思路就是想要简单地移除语料库中的在所有文档中都会出现的单词。通常情 况下，冠词和代词都会被列为停用词。这些单词在一些 NPL 任务(如说关于信息的检索和 分类的任务)中是毫无意义的，这意味着这些单词通常不会产生很大的歧义。

恰恰相反的 是，在某些 NPL 应用中，停用词被移除之后所产生的影响实际上是非常小的。在大多数时 候，给定语言的停用词列表都是一份通过人工制定的、跨语料库的、针对最常见单词的停 用词列表。

虽然大多数语言的停用词列表都可以在相关网站上被找到，但也有一些停用词 列表是基于给定语料库来自动生成的。

有一种非常简单的方式就是基于相关单词在文档中 出现的频率(即该单词在文档中出现的次数)来构建一个停用词列表，出现在这些语料库 中的单词都会被当作停用词。

经过这样的充分研究，我们就会得到针对某些特定语料库的 最佳停用词列表。

NLTK 库中就内置了涵盖 22 种语言的停用词列表。

## 罕见词移除

这是一个非常直观的操作，因为该操作针对的单词都有很强的唯一性，如说名称、品 牌、产品名称、某些噪音性字符(例如 html 代码的左缩进)等。

这些词汇也都需要根据不同的 NLP 任务来进行清除。

例如对于文本分类问题来说，对名词的使用执行预测是个很坏 的想法，即使这些词汇在预测中有明确的意义。我们会在后面的章节进一步讨论这个问题。 

总而言之，我们绝对不希望看到所有噪音性质的分词出现。为此，我们通常会为单词设置 一个标准长度，那些太短或太长的单词将会被移除:

# 文本分类

对于文本分类，最简单的定义就是要基于文本内容来对其进行分类。

通常情况下，目前所有的机器学习方法和算法都是根据数字/变量特征来编写的。所以这里最重要的问题之一，就是如何在语料库中用数字特征的形式来表示文本。

## 取样操作

一旦以列表的形式持有了整个语料库，接下来就要对其进行某种形式的取样操作。 通常来说，对语料库的整体取样方式与训练集、开发测试集和测试集的取样方式是类似的，整个练习背后的思路是要避免训练过度。

如果将所有数据点都反馈给该模型， 那么算法就会基于整个语料库来进行机器学习，但这些算法在真实测试中针对的是不可 数据。

在非常简单的词汇环境中，如果在模型学习过程中使用的是全体数据，那么尽管分 类器在该数据上能得到很好的执行，但其结果是不稳健的。原因在于一直只在给定数据上执行出最佳结果，但这样它是学不会如何处理未知数据的。

## 词汇文档矩阵(term-document matrix) & 词袋 BOW(bag of word)

整个文本转换成向量形式。

文本文档也可以用所谓的 BOW(bag of word)来表示，这也是文本挖掘和其他相 关应用中最常见的表示方法之一。基本上，不必去考虑这些单词在相关语境下的表示方式。



# 分类器

## 朴素贝叶斯法

依赖于贝叶斯算法，它本质上是一个根据给定特征/属性，基于某种条件概率为样本赋予某
个类别标签的模型。在这里，将用频率/伯努利数来预估先验概率和后验概率。 

$$ 后验概率= \frac{先验概率 × 似然函数}{证据因子} $$

朴素算法往往会假设其中所有的特征都是相互独立的，这样对于文本环境来说看起来会直观一些。

但令人惊讶的是，朴素贝叶斯算法在大多数实际用例中的表现也相当良好。

朴素贝叶斯(NB)法的另一个伟大之处在于它非常简单，实现起来很容易，评分也很简单。只需要将各频率值存储起来，并计算出概率。无论在训练时还是测试(评分)时， 它的速度都很快。基于以上原因，大多数的文本分类问题都会用它来做基准。

## 决策树

决策树是最古老的预测建模技术之一，对于给定的特征和目标，基于该技术的算法会 尝试构建一个相应的逻辑树。使用决策树的算法有很多种类，这里主要介绍的是其中最着 名和使用最广泛的算法之一:*CART*。

CART 算法会利用特性来构造一些二叉树结构，并构造出一个阈值，用于从每个节点 中产生大量的信息。

## 随机梯度下降法

随机梯度下降(Stochastic gradient descent，简称 SGD)法是一种既简单又非常有效 的、适用于线性模型的方法。

尤其在目标样本数量(和特征数量)非常庞大时，其作用会特别突出。如果参照之前的功能列表图，我们会发现 SGD 是许多文本分类问题的一站式解 决方案。另外，由于它也能照顾到规范化问题并可以提供不同的损失函数，所以对于线性 模型的实验工作来说它也是个很好的选择。

SGD 算法有时候也被称为最大熵(Maximum entropy，简称 MaxEnt)算法，它会用 不同的(坡面)损失函数(loss function)和惩罚机制来适配针对分类问题与回归问题的线性模型。

例如当 loss = log 时，它适配的是一个对数回归模型，而当 loss = hinge 时，它适 配的则是一个线性的支持向量机(SVM)。

## 逻辑回归

逻辑回归(logistic regression)是一种针对分类问题的线性模型。它在某些文献中也 被称为対元逻辑(logit regression)、最大熵(MaxEnt)分类法或对数线性分类器。在这 个模型中，我们会用一个対元函数来进行建模，以概率的方式来描述单项试验的可能 结果。

## 支持向量机

支持向量机(Support vector machine，简称 SVM)是目前在机器学习领域中最为先 进的算法。

SVM 属于非概率分类器。SVM 会在无限维空间中构造出一组超平面，它可被应用在 分类、回归或其他任务中。

直观来说，可以通过一个超平面来实现良好的分类划界，这个 超平面应该距离最接近训练数据点的那些类最远(这个距离被称为功能边界)，因为在一般 情况下，这个边界越大，分类器的规模就越小。

## 随机森林算法

随机森林是一种以不同决策树组合为基础来进行评估的合成型分类器。

事实上，它比较适 合用于在各种数据集的子样本上构建多决策树型的分类器。另外，该森林中的每个树结构都建立 在一个随机的最佳特征子集上。最后，启用这些树结构的动作也找出了所有随机特征子集中的最 佳子集。总而言之，随机森林是当前众多分类算法中表现最佳的算法之一

# 示例 饭店评论


```python
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)
dataset.info()
dataset.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 2 columns):
    Review    1000 non-null object
    Liked     1000 non-null int64
    dtypes: int64(1), object(1)
    memory usage: 15.7+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Liked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Refresher for regular expressions:



```
'^[a-zA-Z]':  match all strings that start with a letter
'[^a-zA-Z]':  match all strings that contain a non-letter
```


```python
# Cleaning the texts

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
ps = PorterStemmer()
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/iosdevlog/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
corpus[:10]
```




    ['wow love place',
     'crust good',
     'tasti textur nasti',
     'stop late may bank holiday rick steve recommend love',
     'select menu great price',
     'get angri want damn pho',
     'honeslti tast fresh',
     'potato like rubber could tell made ahead time kept warmer',
     'fri great',
     'great touch']




```python
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
#create a feature matrix out of the most 1500 frequent words:
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
X[:5]
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)




```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train[:5]
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)




```python
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
# Predicting the Test set results
# Looking at first 5 testing data, we can see we predicted the first 3 incorrectly as positive reviews, and last 2 correctly as negative review

y_pred = classifier.predict(X_test)
print(y_pred[:5])
print(y_test[:5])
print(cv.inverse_transform(X_test[:5]))
```

    [1 1 1 0 0]
    [0 0 0 0 0]
    [array(['aw', 'food', 'present'], dtype='<U17'), array(['food', 'servic', 'worst'], dtype='<U17'), array(['dine', 'never', 'place'], dtype='<U17'), array(['disgrac', 'guess', 'mayb', 'night', 'went'], dtype='<U17'), array(['avoid', 'lover', 'mean', 'place', 'sushi'], dtype='<U17')]



```python
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
```




    array([[55, 42],
           [12, 91]])



> 参考资料：《NLTK基础教程》，《机器学习 A-Z》


```python

```
