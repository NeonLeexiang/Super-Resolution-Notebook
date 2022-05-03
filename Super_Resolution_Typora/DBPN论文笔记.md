

# DBPN: Deep Back-Projection Networks For SuperResolution

---

**GitHub:** 👉 [alterzero/**DBPN-Pytorch**](https://github.com/alterzero/DBPN-Pytorch)

**Paper:**👉 [Deep Back-Projection Networks for Single Image Super-resolution](https://arxiv.org/abs/1904.05677)

---

## 来自知乎的论文笔记

### 贡献

- （1）Error feedback 错误反馈 我们提出了一个迭代的错误反馈机制，计算up-and down-projection errors上下投影误差来重构以获得更好结果。这里投影误差用来约束前几层的特征。（文中说这是一个自我校正的过程）

- （2）Mutually connected up- and down-sampling stages相互连接的上-下采样阶段 前馈结构可视为一种映射，仅仅将输入的代表性特征到输出空间。这种方法对于LR到HR的映射是不成功的，尤其是大尺度因子，这是因为LR空间的特征局限性。因此我们的网络不仅利用上采样层生成多样的HR特征并且利用下采样层将其映射到LR空间。这种连接在上图有show。这种交替式的上（蓝色box）下（金色box）采样过程表征了LR和HR图像的相互关系。

- （3）Deep concatenation深度级联 我们的网络表示了不同类型的图像退化和HR成分。这种能力使得网络可以利用HR特征图的深度级联来重构HR图像。不像其他网络，我们的重构直接利用不同类型的LR-HR特征，无需在采样层中传播，上图中红色箭头。

- (4) Improvement with dense connection稠密链接实现提升 我们在每个上-下采样阶段利用稠密链接（论文DenseNet：Densely Connected Convolutional Networks）来鼓励特征重用以提升网络精度。

### Projection Units





#### 上采样投影单元





#### 下采样投影单元





### Dense projection units





### Network architecture

有3个部分：

1.Initial feature extraction.初始化特征提取，前面的m个 3*3 卷积和k个1*1卷积，1*1卷积主要是用来降低维度，上下采样单元的卷积核个数相同

2、Back-projection stages 随后的初始特征提取是一系列的反射单元。交替LR和HR特征图 ![[公式]](https://www.zhihu.com/equation?tex=H%5E%7Bt%7D)和 ![[公式]](https://www.zhihu.com/equation?tex=L%5E%7Bt%7D) 的构建，每个单元可以接触到所有之前单元的输出。

3.Reconstruction 最后的重建过程，如图中红线所示，把所有上采样单元的输出concat一起，然后经过卷积后输出HR。

### 读后感

这篇文章主要是提出交替迭代的上采用和下采样单元，反向传播投影误差，在网络中多次校正重建结果。上采样单元生成更多的HR特征，下采样单元把这些特征投影到LR空间，这样可以保留更多的HR成分，同时产生更多的的深度特征用于重建HR和LR特征。

---

## 论文内容