# 线性回归

目标:找到一个$f(w) = w^Tx$,然后能够拟合数据样本。

## 最小二乘估计(LSE)

### 两个角度理解最小二乘法的意义

#### 距离角度的理解

找到一个超平面，然后使得所有的样本距离这个超平面的距离之和最短。

#### 向量空间角度的理解

我们假设不同的样本（N个样本）构成了一个N维度的向量空间，然后样本标签Y是一个不在这个向量空间中的一个向量，然后需要找到一个线性组合$\beta$然后使得$X\beta$之后形成的新的向量空间和Y距离最近，模型可以写成 $f(w)=X\beta$，于是它们的差应该与这个张成的空间垂直。
$$
X^T(Y-X\beta) = \vec{0} \\
\rightarrow 
\beta = (X^TX)^{-1}X^TY
$$

### LSE


$$
L(w) = \sum_{i=1}^N||w^Tx_i - y_i||^2\\
= \sum_{i=1}^T(w^Tx_i - y_i)^2\\
= (w^TX^T-Y^T)(XW-Y)\\
= w^TX^TXw - 2w^TX^TY + Y^TY
$$
然后，针对于$w$，可以这样计算
$$
\hat{w} = \arg \min L(w) 
$$
对$w$做偏导数运算
$$
\frac{\partial{L(w)}}{\partial(w)} = 2X^TXw -2X^TY = 0
$$
求得
$$
w = (X^TX)^{-1}X^TY
$$

### 从概率的角度看LSE

前提条件,$\epsilon$是噪声
$$
\epsilon \sim N(0,\sigma^2) \\ 
y = f(w) + \epsilon\\
f(w) = w^T \\
y = w^T+\epsilon\\
y|x_iw\sim N(w^Tx,\sigma^2)
$$
得到MLE也就是最大似然估计
$$
L(w) = \log p(Y|(X;w))\\
= \sum_{i=1}^N(\log{\frac{1}{\sqrt{2\pi}\sigma}} - \frac{1}{2\sigma^2}(y_i-w^tx_i)^2)
$$
优化函数变为:
$$
\hat{w} = \arg \max_{w} L(w) \\
= \arg \min_w(y_i-w^tx_i)^2
$$
看最终结果，结果又转换到最小二乘的基本表达形式了。

## 正则化

正则化框架$L(w)$是损失函数,$p(w)$是惩罚项。
$$
\arg \min_{w} [L(w) +\lambda p(w)]
$$


### L1正则化



$L_1 :Lasso , p(w) = ||w||_1$



### L2正则化

$L_2:Ridge 岭回归 p(w) = ||w||_2 = w^Tw$

#### 概率派看L2正则化

然后损失函数变为:
$$
J(w) = \sum_{i=1}^N||w^Tx_i - y_i||^2 + \lambda w^Tw \\
= w^T(X^TX+\lambda I)w -2w^TX^TY+Y^TY
$$
最优化函数变成：
$$
\hat{w} = \arg \min_{w}J(w)
$$
针对于上述的优化函数，对w求偏导得
$$
\frac{\partial{J(w)}}{\partial{w}} = 2(X^TX+\lambda I)w -2X^TY = 0 \\
\rightarrow \hat{w} = (X^TX+\lambda I)^{-1}X^TY
$$


那么为什么能够降低过拟合呢？

注意我们的参数$\lambda$，如果它比较大，那要想$J(w)$取小值，那么系数$w$就必须减小，这就降低了模型的复杂度，过拟合现象得以缓解。但$\lambda$也不能过大，过大会导致系数被“惩罚”得很厉害，模型反而会过于简单，可能欠拟合；同时，$\lambda$也不能过小，当λ趋近于0的时候，相当于我们没有添加正则化项，同样不能缓解过拟合。

#### 从贝叶斯角度看L2正则化

贝叶斯角度假设参数$w$服从高斯分布$w \sim N(0,\sigma^2_2)$,$\epsilon$是噪声点。
$$
f(w) = w^tx\\
y = f(w) + \epsilon = w^Tx+\epsilon \\
\epsilon \sim N(0,\sigma^2)\\
w \sim N(0,\sigma^2_2) \\
y|x_iw \sim N(w^Tx,\sigma^2)\\
p(w|y) = \frac{p(y|w) \cdot p(w)}{p(y)}
$$
问题转化成为求$p(y|w)$与$p(w)$
$$
p(y|w) = \frac{1}{\sqrt{2\pi}\sigma}\exp\{-\frac{(y-w^Tx)^2}{2\sigma^2}\}\\
p(w) = \frac{1}{\sqrt{2\pi}\sigma_2}\exp\{-\frac{||w||^2}{2\sigma_2^2}\}
$$
最优化问题转化成为:
$$
MAP:\hat{w} = \arg \max_{w} p(w|y) \\
= \arg \min \sum_{i=1}^N(y_i-w^Tx_i)^2 +\frac{\sigma^2}{\sigma_2^2}||w||^2_2
$$
然后$\frac{\sigma^2}{\sigma_2^2}$就相当于正则化中的惩罚项。由此可见贝叶斯角度得到的最大后验概率就等于带有L2正则化的最小二乘估计。

## Conclusion

$$
LSE == MLE(极大似然估计) noise服从高斯分布\\
正则化的LSE == MAP(最大后验概率) noise服从高斯分布同时先验概率 p(w)也服从高斯分布
$$

