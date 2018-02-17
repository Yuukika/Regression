
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

回归问题一般可以归纳为公式：


$$f(x) = w^Tx$$

x为n*m维的样本，w为m*1的参数，y为m*1的目标值，
errors = f(x) – y 为所有样本的预测值和目标值的误差和。
可以定义均方根误差：


$MSE(x,w) = \frac{1}{n}*\sum_{i=1}^n (w^Tx_i - y_i)^2$

向量式为：

$MSE(x,w) = \frac{1}{n}*(w^Tx-y)*(w^Tx-y)^T$

目标是求解当MSE最小时w的值。
求解w有两种方法：
 1. 正规方程
    对MSE求w的导数然后求解等于0时的解：
    $w = (x^Tx)^-1x^Ty$
 2. 梯度下降
    梯度下降的核心思想是通过迭代的方式调整w参数来使MSE函数最小化


对MSE求w的导数可以得到
$ w_j = \frac{2}{n}*\sum_{i=1}^n (w^Tx_i - y_i)*x^j$

这里定义一个学习率α，于是得到:
                 Repeat [
                    $w_j = w_j - \frac{2}{n}*\alpha*\sum_{i=1}^n (w^Tx_i - y_i)*x^j$
                    (同时更新$w_j$,
                    for j = 0,1,...m)
                ]

向量形式：

$w = w - \frac{2}{n}\alpha*x^T(w^Tx - y)$

这里的α决定了沿着让MSE下降程度最大方向的幅度有多大，α过小会让梯度下降算法花费更多的时间才能聚合，而太大的话可能会让MSE越过最低点，甚至无法收敛，导致发散。


还可以设置一个阈值tolerance,当梯度向量减小到这个阈值以下时就可以判定为梯度下降算法收敛了。
以上是批量梯度下降算法。


除了批量的，还有随机梯度算法，批量梯度下降算法每次更新参数时都要计算整个数据集，当数据集很大时，将会变得很慢，随机梯度算法每步只用一个随机样本来计算梯度值更行参数，还利用learning schedule来调整学习率α进而促使收敛。
还有一种叫小批量梯度下降算法，介于上面两种之间，每次随机选取小部分样本进行计算。
·

梯度下降算法的快速实现:

```
alpha = 0.01
num_iters = 400

theta = np.zeros((2,1))
for i in np.arange(num_iters):
    theta = theta - 2 * alpha / n * (X.T).dot(X.dot(theta) - y)
```

随机梯度算法的快速实现：
```
num_iters = 50
t0,t1 = 5,50
def learning_schedule(t):
    return to / (t0 + t1)

theta = np.zeros((2,1))
for iter in range(num_iters):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        alpha = learning_schedule(iter*m + i)
        theta = theta - 2 * alpha / n * (X.T).dot(X.dot(theta) - y)
```

回归的泛化误差包括偏差，方差和噪声之和。
假定我们对一组测试集x进行训练，令$y_D$为数据点的观察目标值，y为数据点的真实值，f(x)为训练集的预测值，$\overline f(x)$为训练模型的平均预测值。可以对误差进行拆解可以得到:
$\epsilon^2 = E_D[y_D - y]^2$
$bias^2(x) = (\overline f(x) - y)^2$
var(x) = $E_D[f(x) - \overline f(x)]^2$
噪声为不可减少误差，偏差值为期望输出与真实标记的误差，方差为预测值的方差
可以分析得到，当模型复杂度较低时，模型的拟合度不够，导致偏差很高而方差很小，但是当模型的复杂度很高时，模型和数据的拟合度很高，模型的平均预测也和原数据很拟合，因此偏差很低但是方差很大。
方差度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响。
偏差度量了学习算法的期望预测和真实结果的偏离程度，刻画了学习算法本身的拟合能力。


增加模型的复杂度会提高模型的variance而降低其bias。
一般情况下通过交叉验证方式对模型的性能进行判断。

交叉验证法将数据集均分成k个大小相同的子集，每次选取一个子集作为测试集，其他k-1个子集作为训练集，一共有k组训练\测试集，可以进行k次训练测试，返回k个测试结果。这种验证方法也称为“K折交叉验证”。
通过交叉验证方式，如果训练的模型在预测训练集时误差低但是在测试集上误差很高则表明模型本身过拟合了。而在训练集和测试集上都变现为高误差时则表明模型的复杂度太低，模型的训练不足。
另一种判断模型性能的方式叫做学习曲线，通过改变训练集样本数，逐步增加训练集样本数，使得模型的训练强度逐渐增强，这时候观察训练集和交叉集的RMSE变化可以判断模型处于什么状态。
当训练样本很少时，模型可以很好的拟合数据，训练集的误差小，但随着训练的样本增多，误差会增大，直至稳定，之后再增加样本也不会产生很大变化。在交叉集上，一开始，训练样本少，模型训练不足，模型的拟合能力不强，对交叉集的预测误差会很大，但是随着训练样本的增加，拟合能力加强，误差逐渐降低，同样也会趋于稳定，此时更多的样本训练也不会起很大作用。在欠拟合模型中，训练集和交叉集稳定后的误差都很高。而在过拟合模型中，训练集和交叉集的误差要低很多，两者之间有个间隔。


对于过拟合问题，可以通过限制模型属性权重的正则化式子来减少。在原来开销函数MSE基础上增加正则化式子，产生了三种典型的正则化模型：Ridge,Lasso,Elastic Net。
在原来MSE基础上增加参数权重向量的l2范数，我们就可以得到岭回归模型：
$J(w) = MSE(w) + \frac{\alpha}{2}*\sum_{i=1}^m w_i^2$

$\alpha因子控制模型的正则化程度，当\alpha$=0时，无正则化，当\alpha非常大时，所有的属性参数都趋近于0，模型曲线变成一条平直线。


Lasso回归：
$J(w) = MSE(w) + \alpha*\sum_{i=1}^m |w_i|$

Lasso回归一个重要的特点是他可以消除一些不重要属性值，也就是说自动进行属性选择，返回一个稀疏模型。


Elastic Net 介于两者之间，正则化式子结合了Ridge和Lasso回归：
$J(w) = MSE(w)+ r\alpha\sum_{i=1}^n|w_i|+ \frac{1-r}{2}\alpha*\sum_{i=1}^nw_i^2$

sklearn上有对应的模块实现了回归算法。
```
from sklearn.linear_model import LinearRegression,Ridge,Lasso
```

```
LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
```
fit_intercept是否计算截距
normalize如果计算截距，那么为正时会在训练前正规化数据
有几个重要的方法：
fit(X,y)训练模型
predict(X)返回预测值
score(X,y)返回想X,y的预测值的R^2
模型的参数包含:
.coef_属性的参数
.intercept_截距的参数


```
linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver=’auto’, random_state=None)
```
alpha正则化系数
fit_intercept与normalize都与之前一样
max_iter:最大迭代次数
tol:模型的精确度
solver:模型的收敛计算方法,包括：{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}
类似于上面的梯度下降。
random_state:随机数种子，有随机选择时，如果随机种子一致能保证每次结果的一致性。
方法与之前的一致
参数多了一个：
.n_iter_实际模型聚合时迭代的次数。



```
linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection=’cyclic’)
```
alpha是L1正则化系数
precompute是否使用预计算Gram矩阵来加速计算。
warm_start为 True 时, 重复使用上一次学习作为初始化，否则直接清除上次方案。
positive设为 True 时，强制使系数为正。
selection : str, 默认 ‘cyclic’
若设为 ‘random’, 每次循环会随机更新参数，而按照默认设置则会依次更新。设为随机通常会极大地加速交点（convergence）的产生，尤其是 tol 比 1e-4 大的情况下。

```
linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection=’cyclic’)[source]
```
alpha和l1_ratio分别对应cost函数中的$\alpha和r$
