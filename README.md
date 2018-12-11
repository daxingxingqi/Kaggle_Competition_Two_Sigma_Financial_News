# 优达学院Kaggle竞赛

# 竞赛题目 Two Sigma: Using News to Predict Stock Movements

地址：https://www.kaggle.com/c/two-sigma-financial-news

Can we use the content of news analytics to predict stock price performance? The ubiquity of data today enables investors at any scale to make better investment decisions. The challenge is ingesting and interpreting the data to determine which data is useful, finding the signal in this sea of information. [Two Sigma ](http://www.twosigma.com/)is passionate about this challenge and is excited to share it with the Kaggle community.

As a scientifically driven investment manager, Two Sigma has been applying technology and data science to financial forecasts for over 17 years. Their pioneering advances in big data, AI, and machine learning have pushed the investment industry forward. Now, they're eager to engage with Kagglers in this continuing pursuit of innovation.

By analyzing news data to predict stock prices, Kagglers have a unique opportunity to advance the state of research in understanding the predictive power of the news. This power, if harnessed, could help predict financial outcomes and generate significant economic impact all over the world.

Data for this competition comes from the following sources:

- Market data provided by Intrinio.
- News data provided by Thomson Reuters. Copyright ©, Thomson Reuters, 2017. All Rights Reserved. Use, duplication, or sale of this service, or data contained herein, except as described in the Competition Rules, is strictly prohibited.

# 参考资料

- [时间序列模型 - 也谈其在计量经济学中的应用1](https://zhuanlan.zhihu.com/p/46347425)



- [时间序列模型 - 也谈其在计量经济学中的应用2](https://zhuanlan.zhihu.com/p/48165114)

# 竞赛规则

以下摘取几个比较重要的规则，全部规则可以后面链接查看https://www.kaggle.com/c/two-sigma-financial-news/rules。

## No private sharing outside teams

Privately sharing code or data outside of teams is not permitted. It's okay to share code if made available to all participants on the forums.

## Team Mergers

Team mergers are allowed and can be performed by the team leader. In order to merge, the combined team must have a total submission count less than or equal to the maximum allowed as of the merge date. `The maximum allowed is the number of submissions per day multiplied by the number of days the competition has been running`.

## Team Limits

The maximum team size is 3.

## Submission Limits

You may submit a maximum of 5 entry submissions per day.

You may select up to 2 final submissions for judging.

## Competition Timeline

- Start Date: **September 25, 2018**
- Entry Deadline: **January 2, 2019**
- Team Merger Deadline: **January 2, 2019**
- Submission Deadline: **January 8, 2019**
- End Date: **July 15, 2019**

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.



## SUBMISSIONS

'Submission' means the material created and submitted by you in the manner and format specified on the Website via the Submission function on the Website. In this Competition, the Submission will be in the form of code. This is a skills-based competition and all submissions must be original code created by you. You (or if you are part of a Team, your Team) may submit up to the maximum number of Submissions per day as specified above. All Submissions must be run on the Website in the manner and format specified on the Website. Submissions must be received prior to the Competition deadline and adhere to the guidelines for Submissions specified on the Website.

Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.

Submissions may not make reference to forward-looking data sources (i.e., any data sources that look ahead in time past the data provided) or make use of external or other non-permitted data. Submissions may not include any material, non-public information.

If the Competition is a multi-stage Competition with temporally separate training data and/or leaderboard data, one or more valid Submissions must be made and selected during each stage of the Competition in the manner described on the Competition Website.

# 竞赛指导

NB：竞赛kernel运行cpu-6hrs，gpu-2hrs

- 官方starter kenerl：https://www.kaggle.com/dster/two-sigma-news-official-getting-started-kernel

  > 仔细研读，很多的竞赛数据的细节和介绍，还有关于如何提交的步骤

- 很不错的探索性分析和基础特征工程kernel：https://www.kaggle.com/artgor/eda-feature-engineering-and-everything

- 如何构建lag特征（历史滑窗）并使用多进程优化速度：https://www.kaggle.com/qqgeogor/eda-script-67

## 机器学习基本步骤

### 1. 特征观察
对数据集进行一个粗略的探索，一般进行如下操作：

a.**观察前五行数据和数据统计学特征**

 `data.head()`

`test.describe()`

b.**观察数据的整体情况（类型，数量，特征，有无缺失值等）和缺失值的整体数量**

`data.info()`

`test.isnull().sum().sort_values(ascending = False)`

c.**结合数据情况和要求观察数据**

注意事项：

>   - 标签是否balance
    - 理解数据中包含哪些列（数值，类目特征）
    - 数据量（train,test分布）
    - 缺失数据与离群值

### 2. 准备数据（数据清洗，格式化和重新组织）

在数据能够被作为输入提供给机器学习算法之前，它经常需要被清洗，格式化，和重新组织 - 这通常被叫做**预处理**。一般进行如下操作：

a.**获得特征和标签**

Example 来自于 finding_donor（https://github.com/daxingxingqi/Supervised_learning/blob/master/finding_donors/finding_donors.ipynb）

```python
# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
```

b.**离群点检测**

通关观察数据，并结合相应知识，把离群点剔除

Example 来自于 ameshouse（https://github.com/daxingxingqi/CSDN_Homework/tree/master/CSDN-homework-1/Homework_2）

```python
# 离群点检测（outliers）
plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

#剔除离群点
train = train[train.GrLivArea < 4000]
temp = train.reindex()
```

c.**缺失值补充**

- 数值特征
  - median/mean 

  - most common value 

Example 来自于 ameshouse（https://github.com/daxingxingqi/CSDN_Homework/tree/master/CSDN-homework-1/Homework_2）
```python
numerical_features = df.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
df.info()
df_num = df[numerical_features]
#df_num.info()
medians = df_num.median() 
# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in df : " + str(df_num.isnull().values.sum()))
df_num = df_num.fillna(medians)
print("Remaining NAs for numerical features in df : " + str(df_num.isnull().values.sum()))
```

- 非数值特征（可以结合数据情况填充）

  Example 来自于 ameshouse（https://github.com/daxingxingqi/CSDN_Homework/tree/master/CSDN-homework-1/Homework_2）
```python
# Alley : data description says NA means "no alley access"
    df.loc[:, "Alley"] = df.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
    df.loc[:, "BedroomAbvGr"] = df.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
    df.loc[:, "BsmtQual"] = df.loc[:, "BsmtQual"].fillna("No")
    df.loc[:, "BsmtCond"] = df.loc[:, "BsmtCond"].fillna("No")
    df.loc[:, "BsmtExposure"] = df.loc[:, "BsmtExposure"].fillna("No")
    df.loc[:, "BsmtFinType1"] = df.loc[:, "BsmtFinType1"].fillna("No")
    df.loc[:, "BsmtFinType2"] = df.loc[:, "BsmtFinType2"].fillna("No")
    df.loc[:, "BsmtFullBath"] = df.loc[:, "BsmtFullBath"].fillna(0)
    df.loc[:, "BsmtHalfBath"] = df.loc[:, "BsmtHalfBath"].fillna(0)
    df.loc[:, "BsmtUnfSF"] = df.loc[:, "BsmtUnfSF"].fillna(0)

# CentralAir : NA most likely means No
    df.loc[:, "CentralAir"] = df.loc[:, "CentralAir"].fillna("N")
 # Condition : NA most likely means Normal，靠近主干道或铁路
    df.loc[:, "Condition1"] = df.loc[:, "Condition1"].fillna("Norm")
    df.loc[:, "Condition2"] = df.loc[:, "Condition2"].fillna("Norm")

```

d.**规一化数字特征**

<div align=center><img width="450" src=resource/scaling.png></div>

除了对于高度倾斜的特征施加转换，对数值特征施加一些形式的缩放通常会是一个好的习惯。在数据上面施加一个缩放并不会改变数据分布的形式；但是，规一化保证了每一个特征在使用监督学习器的时候能够被平等的对待。注意一旦使用了缩放，观察数据的原始形式不再具有它本来的意义了。使用[`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)。

- Standard Scaler

  使data分布变为mean为0，std为1

```python
import pandas pd
from sklearn import preprocessing

# standardise the means to 0 and standard error to 1
for i in df.columns[:-1]: # df.columns[:-1] = dataframe for all features
  df[i] = preprocessing.scale(df[i].astype('float64'))
```

- Min Max Scale

  使所有数据位于0到1

<div align=center><img width="200"src=resource/minmaxscaler.png></div>

```python
import pandas pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
```
- RobustScaler
  - 与 standard scaler 类似，但是RobustScaler使用 median 和 quartiles，而不是mean 和 varience。这样就会避免离群点的影响

- Normalizer
   - 使得所有的data的欧式距离为1
   
- Pipeline
>Scaling有可能会造成数据的leak，**非常建议使用pipeline**，理由如下：
>
>- Scaling have a chance of leaking the part of the test data in train-test split into the training data. This is especially inevitable when using cross-validation. A way to prevent data-leakage is to use the pipeline function in sklearn, which wraps the scaler and classifier together, and scale them separately during cross validation.

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])

pipe.fit(X_train, y_train)
Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svm', SVC(C=1.0, cac
          decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
          max_iter=-1, probability=False, random_state=None, shrinking=True,
          tol=0.001, verbose=False))])

pipe.score(X_test, y_test)
0.95104895104895104
```



注意事项：

>   - 去除缺失值，缺失值填充，修正标注缺陷
>   - Scale和Normalization
>   - 时序数据分类
>   - 同时涉及到时序特征和普通特征

### 3. 特征工程
a.**数据转换**

> - 线性模型，对于回归标签变换能够起到不错的效果
> - 通过变换使得数据更能符合“正太分布”
> - 常用的转换包括sqrt(x)，sqrt(x+n)，log(x)，log(x+n)，等等
> - 使用哪一种方法需要通过测试，对于一些金融回归数据通常需要对其进行转换，去除异方差

Example 来自于 finding_donor（https://github.com/daxingxingqi/Supervised_learning/blob/master/finding_donors/finding_donors.ipynb）
```python
# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
```

b.**特征编码——类目特征**

>- 大多数情况下，类目特征需要通过编码转换成数值类型特征，才能输入模型中进行训练
>- 特征编码的目的：
>- 显式的处理与标签存在非线性关系的特征（比如颜色，方向等不存在“大小”概念的类目特征）
>- 模型只能接受数值类型特征（大部分sklearn model，xgboost，lightgbm，nn等）
- 1.Label Encode
  - 使用sklearn.preprocessing中的LabelEncoder

  - 直接将类目转换成数字，使用id与原类目一一对应

  - 适用于树模型

<div align=center><img width="450" src=resource/label_encode.png></div>

一般有两种方法如下：

- 字母顺序 `sklearn.preprocessing.LabelEncoder`
- 出现顺序 `pd.factorize`

```python
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x[:, 0] = labelencoder.fit_transform(x[:, 0])
```



- 2.Onehot Encode
  - 使用sklearn.preprocessing中的OneHotEncoder

  - 生成的独立的二进制特征（0或者1）

  - 适用于线性模型，树模型，神经网络，svm，k-means等

<div align=center><img width="450" src=resource/onehot_encode.png></div>

一般有两种方法：

- Dummies: `pd.get_dummies`, this converts a string into binary, and splits the columns according to n categories
- sklearn: `sklearn.preprocessing.OneHotEncoder`, string has to be converted into numeric, then stored in a sparse matrix.

Example 来自于 finding_donor（https://github.com/daxingxingqi/Supervised_learning/blob/master/finding_donors/finding_donors.ipynb）

```python
# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
income = np.where(income_raw == '>50K', 1, 0)

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))

# 移除下面一行的注释以观察编码的特征名字
print(encoded)
```

或者

```python
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
```



- 3.基于频率的特征编码
  - 统计每一个类目特征值出现的频率，作为该类特征的特征值
  - 基于假设：存在相同频率的特征作用于标签具有相同的效果
<div align=center><img width="450" src=resource/frequency_encode.png></div>

```python
### FREQUENCY ENCODING

# size of each category
encoding = titanic.groupby('Embarked').size()
# get frequency of each category
encoding = encoding/len(titanic)
titanic['enc'] = titanic.Embarked.map(encoding)

# if categories have same frequency it can be an issue
# will need to change it to ranked frequency encoding
from scipy.stats import rankdata
```



- 4.基于标签的特征编码
  - 统计每一个类目特征值关于标签的平均值

  - 基于假设：存在相同平均值的特征作用于标签具有相同的效果

  - 需要注意：

    - 基于标签求平均值的编码方式很容易引入leakage，即在训练过程中引入做validation dataset中的信息，造成评估误差，进而使得模型泛化能力变差
<div align=center><img width="450" src=resource/target_mean_encode.png></div>

- 5.基于标签的特征编码——LOO Feature
  Leave-One-Out
  - 针对可能存在leakage的情况，LOO feature在target mean encoding的基础上引入了leave-one-out机制
<div align=center><img width="450" src=resource/loo_encode.png></div> 

c.**特征编码——数值特征**

>-1.可以对数值特征二值化，使用百分比区间，或者利用直方图等方式，将固定区间内的数值特征转化为对应的区间id特征，或者在这基础上利用Onehot encodng方式生成0/1特征，在某些情况下可以缓解过拟合情况的发生
>- 2.通过SVD，PCA等方式进行降维，提取特征，通常用于特征的线性降维，且数据中存在大量噪音的情况。
>- 3.使用K-means Cluster等聚类方法，获取聚类中心类，或者到每一个聚类中心距离作为新的特征。
>- 4.类似2和3中的方法，可以使用manifold learning或者自编码器中的一些方法（比如tsne、vae等）生成降维特征，这种降维特征通常是原始特征的非线性变换，拥有更强的表征能力

- 特征交叉和组合

  - 将两个已知特征通过一个函数f(x)——可以是线性或者非线性——进行组合，生成一个新的特征，使得原本无法学习到的非线性关系可以被表征出来

    - 例如：

      - 已知特征x1,x2,预测y

      - y是通过函数y=2·(x1^2+x2^3)的方式生成的

      - 给定一个线性回归模型，f(x)=w1·x1+w2·x2+b是难以拟合这种状态的

      - 如果对特征x1,x2进行变换组合，生成一个新的特征x3 = x1^2+x2^3，则只需要拟合f(x) = 2·x3+b，从而使得线性模型完美解决问题

  - 如何寻找特征交叉和组合
    - 领域知识：比如已知某个租房租金，且已知租房中有多少个房间，预测住房的热租程度，很明显可以看出平均每个房间的价格会直接影响到房间的受欢迎程度
    - 遗传算法：通过遗传算法尝试各种组合，挑选其中最合适的（效率通常低下，但有时候很有效）
    - 通过模型权重或者参数反馈（比如分析树模型的split点）
  - 如何构建特征交叉和组合
    - 聚类特征
    - 基于标签的编码
    - 基于其他数值特征的类目特征编码
d.**特征编码——时序特征处理**
>- 对于一些数据集来说，例如这一次的twosigma给出的是时序相关的数据可以构造lag特征：
>- 马尔科夫假设：下一个时间点的状态只和前N个时间点相关，而和N以前的时间点无关
>- 检验时序数据是否满足马尔科夫假设，如果是，则没有必要使用所有历史信息构建特征
>- 基于马尔科夫假设构建滑窗，构建每一个window size大小内的特征，构建方法包括但不限于求这段时间内的min,max,mean,median,std等数值

注意事项：

>  - 寻找新的gold feature，去除有噪音或者不相关的特征
>   - 通过检验特征重要性，或者通过各类特征选择手段
>     - 1.基于模型权重
>     - 2.基于统计方法
>     - 3.基于模型验证
>  - 特征工程是成功的机器学习项目中最重要的因素之一
>   - 通常情况下是特征而非模型决定项目的上限
>  - 特征工程区分：
>     - 标签转换/target transformation
>     - 特征编码/feature encoding
>     - 特征抽取/feature extraction

e.**特征提取**

>- 部分特征中包含隐含信息，可以被提取成新的特征
>- 时间：年月日——》提取周，季度，是否为工作日，是否为假期等
>- 地点：地址——》街道，门牌号，省份，城市，区块，经纬度等
>- 年龄：各个年龄阶段——》幼儿，少年，青年，中年，老年等

f.**特征提取——文本**
>文本特征的提取有别于传统特征提取，其常采用的方法有：
>
>1.bag-of-word/tfidf，可用sklearn.feature_extraction.text中的CountVectorizer和TfidfVectorizer类达成功能
>
>2.ngram language model
>
>3.各种文本的预处理手段，停用词，词干词缀（英文），大小写处理，标点符号处理等
>
>4.各种pretrianed embedding，例如常用的word2vec，glove，covec等，包括最新google出品的bert等都可以作为feature extractor使用
>
>5.预训练的深度模型，用于提取文本特征
>
>6.基于上述4、5的特征，可以计算文本相似度
>
>7.其他

g.**特征选择**
>在很多时候，特征会有一些冗余带来的特征之间的共线性，或者干脆是一些无效特征，在训练模型的过程中导致一定程度的过拟合
>
>因此在一些场景下需要使用特征提取来处理数据集
>
>- 1.基于统计的方法
>- 2.基于模型参数的方法
>- 3.基于模型训练预测结果的方法
>

- 1.基于统计的方法
  - 常用的包括皮尔森系数，斯皮尔曼系数等
  - 各种统计假设检验的方法也可以用来有目标的选择特征进行训练
- 2.基于模型参数的方法
  - 最常见的Ridge或者Lasso等线性算法，用以去除特征中的冗余
  - 同时树模型也可以用来做特征选择
  - 这些都包含在了sklearn中
  - 可以使用sklearn.feature_selection.SelectFromModel类进行特征提取操作
- 3.基于模型训练结果的方法	
  - 通过策略性的添加，删除数据中的特征，并通过cv验证结果来选择
  - 一般可以使用任何模型来进行训练，但是计算量非常大，谨慎使用
  - 代表方法为sklearn.feature_selection.RFECV，recursive feature elimination
  - 通过cv结果循环去除数据中的特征	

### 4. 模型训练和预测
a.**模型训练**

- 平均

  - 最简单的融合策略，使用多个模型预测的结果取平均值
  - 通常使用这种策略时模型的精度相差不会特别大

- 加权平均

  - 当模型存在一定的精度差距时，可以根据自己的经验和尝试，赋予不同模型预测结果不同的权重，然后加权平均得到最终结果
  - 可以使用验证集来调试权重分配比例，然后再提交，或者可以根据leaderboard的分数反馈调整权重分配比例

- Stacking
  - 在kaggle中常见的stacking融合策略
  - 相比加权平均一般可以得到更好的精度，但是更耗时耗力
  - 一个两层的stacker训练步骤如下：
    - 选择多种模型作为base model，通过oof的方式训练并预测数据，预测预留出的fold的值并拼接生成第一层的metafeature
    - 将所有metafeature合并成一个dataset作为新的特征，交给第二层，由第二层训练并预测最终的结果

- bagging

b.**模型验证**

- 常用的模型验证方法
  - 1fold hold out
  - cross validation
-根据划分fold方式的不同还可以分为
  - random split——sklearn.model_selection.KFold
  - stratified split——sklearn.model_selection.StratifiedKFold 
- 根据时间的划分方式：
  - 时间序列相关的验证划分是特殊的
  - 处理不当很容易将未来的信息加入到training data中，造成leakage
  - 正确的做法是确保一直使用“未来”得数据作为验证集
  - 可以自己指定时间划分train和validation set，也可以使用sklearn中的skelarn.model_selection.TimeSeriesSplit生成时序训练和验证数据集


### 5. 调优

### 6.模型测试
