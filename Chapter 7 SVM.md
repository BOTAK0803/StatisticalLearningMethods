# Chapter 7 SVM

SVM：间隔，对偶，核技巧

## hard-margin SVM

### 将问题转化成为一个具有N个约束的凸优化问题

超平面：$w\cdot x + b = 0$

判别函数：$f(x) = sign(w\cdot x + b)$

最大间隔分类器：$\max margin((w,b))$，找到一个超平面，然后使得距离超平面最近的点的距离达到最大
$$
margin((w,b)) = \min_{w,b,i=1,2,...,N} distance(w,b,x_i)
$$

$$
distance(w,b,x_i) = \frac{1}{||w||}|w\cdot x_i + b|
$$

转化成为：
$$
\max_{w,b}\min_{x_i} \frac{1}{||w||}|w\cdot x_i+b|
$$
去绝对值，因为$y_i(x\cdot x_i + b) >0$
$$
\begin{cases}
max_{w,b}\min_{x_i}\frac{1}{||w||} y_i(w\cdot x_i +b) & \\
s.t. y_i(w\cdot x_i +b) > 0
\end{cases}
$$


肯定$\exists \gamma > 0 ,s.t. \min_{x_i,y_i}y_i(w\cdot x_i + b) = \gamma$进而转化
$$
\begin{cases}
\max_{w,b}\frac{1}{||w||} &\\
s.t. \min_{x_i,y_i}y_i (w\cdot x_i + b) = 1
\end{cases}
$$
继续转化
$$
\begin{cases}
\max_{w,b}\frac{1}{||w||}  = \max_{w,b} \frac{1}{2} w^Tw&\\
s.t. y_i(w\cdot x_i + b) \geq 1 , i = 1,2,...,N
\end{cases}
$$
到此转化成为了一个具有N个约束的凸优化问题.

### 将问题转化成为无约束问题

利用拉格朗日乘子法

原问题:
$$
\begin{cases}
\min_{w,b} \frac{1}{2} w^tw & \\
s.t. 1-y_i(w\cdot x_i + b) \leq 0
\end{cases}
$$
拉格朗日乘子法
$$
L(w,b,\lambda) = \frac{1}{2}w^tw +\sum_{i=1}^n \lambda_i[1-y_i(w\cdot x_i + b)]
$$
转化成为
$$
\begin{cases}
\min_{w,b} \max_{\lambda} L(w,b,\lambda) &\\
s.t. \lambda_i \geq 0
\end{cases}
$$
证明上述转化的正确性,

如果$1-y_i(w\cdot x_i + b) >0$,则$\max_{\lambda}L(w,b,\lambda) = \frac{1}{2}w^tw +\infty = \infty$

如果$1-y_i(w\cdot x_i +b) \leq 0$，则$\max_{\lambda}L(w,b,\lambda) = \frac{1}{2} w^Tw$

综上$\min_{w,b} \max_{\lambda}L(w,b,\lambda) = \min_{w,b}\frac{1}{2}w^Tw$

利用强对偶关系进行转化
$$
\begin{cases}
\max_{\lambda}\min_{w,b} L(w,b,\lambda) &\\
s.t. \lambda_i\geq 0
\end{cases}
$$
求解$\min_{w,b}L(w,b,\lambda)$，针对于$L(w,b,\lambda)$，对$w$和$b$求偏导得：
$$
\begin{cases}
\frac{\partial{L(w,b,\lambda)}}{\partial{b}} = 0 &\\
\frac{\partial{L(w,b,\lambda)}}{\partial{w}} = 0
\end{cases}
$$
求解上面的式子可得
$$
\begin{cases}
\sum_{i=1}^N\lambda_iy_i = 0 &\\
w = \sum_{i=1}^N \lambda_iy_ix_i
\end{cases}
$$
将上述结果反带入$L(w,b,\lambda)$得
$$
L(w,b,\lambda) = -\frac{1}{2} \sum_{i=1}^N\sum_{j=1}^N \lambda_i\lambda_jy_iy_jx_i^Tx_j + \sum_{i=1}^N\lambda_i
$$



$$
\begin{cases}
\max_{\lambda} -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \lambda_i\lambda_jy_iy_jx_i^Tx_j+\sum_{i=1}^N\lambda_i&\\
st. \lambda_i\geq0
\end{cases}
$$
将最大转化为最小有
$$
\begin{cases}
\max_{\lambda} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \lambda_i\lambda_jy_iy_jx_i^Tx_j-\sum_{i=1}^N\lambda_i&\\
st. \lambda_i\geq0 &\\
\sum_{i=1}^N \lambda_iy_i = 0
\end{cases}
$$

### 对偶问题之KKT条件

#### kkt条件

$$
\begin{cases}
\frac{\partial{L}}{\partial{w}} = 0&\\
\frac{\partial{L}}{\partial{b}} = 0&\\
\frac{\partial{L}}{\partial{\lambda}} = 1&\\
\lambda_i[1-y_i(w\cdot x_i + b)] = 0 &\\
\lambda_i\geq 0& \\
1- y_i(w\cdot x +b) \leq 0
\end{cases}
$$

其中第四个条件称之为松弛互补条件，观察KKT条件可以得到，当$1 - y_i(w\cdot x_i+b) ！=0$的时候，$\lambda_i ==0$，没有意义，所以让$1- y_i(w\cdot x_i + b) = 0$。

假设$(x_k,y_k)$是与超平面平行的且最远的间距在此线上的一点，则$1- y_k(w\cdot x_k + b) = 0$，可以求得
$$
\begin{cases}
w^* = \sum_{i=1}^N \lambda_iy_ix_i &\\
b* = y_k - \sum_{i=0}^N \lambda_iy_i^Tx_k
\end{cases}
$$

$$
f(x) = sign(w^*x+b^*)
$$



## soft-margin SVM

在现实问题中，训练数据集往往是线性不可分的，即在样本中出现噪声点或者特异点，soft-margin SVM就是一种解决方案。

soft:允许一点点的错误
$$
\begin{cases}
\min{w,b}\frac{1}{2} + c\sum_{i=1}^N \max\{0,1-y_i(w\cdot x_i + b)\} &\\
st. y_i(w\cdot x_i + b) \geq 1
\end{cases}
$$
引入$\xi_i = 1- y_i(w\cdot x_i + b),\xi_i \geq 0$
$$
\begin{cases}
\min{w,b}\frac{1}{2} + c\sum_{i=1}^N\xi _i&\\
st. y_i(w\cdot x_i + b) \geq 1 -\xi_i
\end{cases}
$$




## kernel SVM

## 题外话

### 对偶关系

$$
\min\max L \geq \max\min L
$$

 有点中国话宁为凤尾不为鸡头的意思 max是凤，min是鸡，当取等号的时候就是强对偶关系

