# PCA

## 序言

​	在模型训练的时候，经常会遇到过拟合的问题，一般而言解决过拟合有很多方法
$$
解决过拟合
\begin{cases}
增大数据量 &\\
正则化 &\\
降维
\begin{cases}
直接降维 (特征选择)&\\
线性降维 
\begin{cases}
PCA &\\
MDS &\\
\end{cases}
&\\
非线性降维
\begin{cases}
流形学习 &\\
ISomap &\\
LLE
\end{cases}
\end{cases}

\end{cases}
$$
​	在模型训练的时候，还存在维度灾难的问题（会导致数据的稀疏性），从几何角度而言，解释维度灾难。

## 均值与方差

​	将样本的均值与方差利用矩阵表示
$$
Data : X = (x_1,x_2,...,x_N)^T_{N*P}
= \left[ \begin{matrix}
x_1^T\\
x_2^t \\
.\\
x_n^T
\end{matrix}
\right]
= \left[
\begin{matrix}
x_{11},x_{12},...,x_{1P}\\
x_{21},x_{22},...,x_{2P}\\
...\\
x_{N1},x_{N2},...,x_{NP}
\end{matrix}
\right]_{N*P}
$$

$$
1_N =
\left[
\begin{matrix}
1\\
1\\
.\\
1
\end{matrix}
\right]_{N*1}
$$
样本X的均值:
$$
\overline{X} = \frac{1}{N}X^T1_N
$$
样本X的方差
$$
H_N = I_N-\frac{1}{N}1_N1_N^T
$$

$$
I_N = \left[
\begin{matrix}
1,0,0,...,0\\
0,1,0,...,0\\
...\\
0,0,0,...,1\\
\end{matrix}
\right]
$$


$$
H^n= H
$$

$$
S = \frac{1}{N}\sum_{i=1}^{N}(x_i - \overline{x})\cdot(x_i - \overline{x})^T = \frac{1}{N}X^THX
$$

## PCA

PCA可以概括成为一个中心，两个基本点

- 一个中心：原始特征空间的重构（例如：学历与学位相似，重构）
- 两个基本点
  - 最大投影方差
  - 最小重构距离

### 最大投影方差

具体而言，用二维空间中的样本举例，在二维空间中有一堆点，现在需要找到一个向量，使得这些到这个方向上面的投影方差最大（越稀疏方差也就越大也就），然后投影到这个方向上面去，假如需要利用PCA将维度降到P维，就选取P个点就可以了。

​	样本点在方向向量$u_1$上的投影可以表示为
$$
\vec{a}\cdot\vec{b} = |\vec{a}|\cdot|\vec{b}|\cdot \cos\theta\\
投影 = |\vec{a}|\cdot \cos \theta = a^Tb\\ 
J = \frac{1}{N}\sum_{i=1}^N((x_i-\overline{x})^Tu_1)^2=\\
u_1^T(\frac{1}{N}\sum_{i=1}^{N}((x_i - \overline{x})\cdot(x_i - \overline{x})^T) u_1 \\
$$


​	那么，现在的目的就转化成为，找到一个方向向量$u_1$，使得满足上面两个基本点
$$
\begin{cases}
\hat{u_1}= \arg \max {u_1^T\cdot S\cdot u_1}&\\
st. u_1^Tu = 1
\end{cases}
$$
 使用拉格朗日乘子法
$$
L(u_1,\lambda) = u_1^T\cdot S\cdot u_1 + \lambda(1-u_1^Tu)
$$
对拉格朗日做偏导数
$$
\frac{\partial{L}}{\partial{u_1}} = 2S\cdot u_1 - \lambda\cdot 2u_1 = 0
$$
得到 
$$
S_{N*N}u_1 = \lambda u_1
$$
由上面的式子可以看出就是求方差矩阵的**特征值**与**特征向量**的过程。



### 最小重构代价

针对于二维样本空间中某一个样本点$x_i$,其在方向$u_1$上的投影重构成为$x_i$的重构代价为$x_i = (x_i^Tu_1 )u_1 + (x_i^Tu_1)u_2$

p维样本空间中某一个样本点的重构代价为$x_i = \sum_{k=1}^p(x_i^Tu_k)u_k$

假设利用PCA将维度降到了q维度，则重构代价为$\hat{x_i} = \sum_{k=1}^q(x_i^Tu_k)u_k$

综合所有的样本，则全部的重构代价为
$$
J = \frac{1}{N}\sum_{i=1}^N||x_i - \hat{x_i}||^2\\
 = \begin{cases}
 \sum_{k=q+1}^p u_k^T\cdot S\cdot u_k&\\
 st. u_k^Tu_k = 1
 \end{cases}
$$
问题转化成为最小化问题,找到一个方向$u_k$使得:
$$
\begin{cases}
u_k = \arg \min \sum_{k=q+1}^p u_k^T\cdot S \cdot u_k &\\
st. u_k^Tu_k = 1
\end{cases}
$$


## PCA SVD角度(PCA PCoA)

​	首先因为方差矩阵$S$是对称矩阵，矩阵的奇异值分解就等同于特征值分解，故
$$
S = GKG^T\\
G^TG=I \\
k = \left[ \begin{matrix} k_1 ...\\.k_2..\\...\\..k_p
\end{matrix}\right]
$$
直接对样本进行分析,首先进行中心化 $HX$,然后进行奇异值分解
$$
HX = U\Sigma V^T \\
st.\begin{cases}
H^T=H\\
H^n=H\\
U^TU=I\\
V^TV = VV^T = I\\
\Sigma 是对角矩阵
\end{cases}
$$
然后看对方差矩阵的分解
$$
S_{p*p} = X^THX = X^TH^THX = V\Sigma U^T \cdot U\Sigma V^T = V\Sigma^2V^T  
$$
其中$V$是特征向量，$k = \Sigma^2$是特征值矩阵

#### PCoA(主坐标分析 principle coordinate analysis)

​	找到数据中最主要的样本坐标
$$
T_{N*N} = HXX^TH = U\Sigma V^T\cdot V\Sigma U^T = U\Sigma^2U^T
$$
T和S有相同的特征值

S特征分解，分解到方向上面，也就是主成分上面去，然后$HX\cdot V$转换成为坐标
$$
HX\cdot V = U\Sigma V^TT = U\Sigma \\
T = U\Sigma^2U^T =>\\
TU\Sigma = U\Sigma^2U^TU\Sigma = U\Sigma^3 = U\Sigma\Sigma^2
$$
其中$U\Sigma$是特征向量组成的矩阵，$\Sigma^2$是特征值矩阵

T特征分解直接分解到坐标上面去



## PCA 概率角度(P-PCA)



