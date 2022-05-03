# BasicVSR

[BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond](https://arxiv.org/pdf/2012.02181.pdf)

## 论文内容

#### Abstract

由于需要利用额外的时序信息，视频超分往往比图像超分包含更多的模块，这就导致了各式各样的复杂结构。

 该文作者对视频超分进行了梳理并重新审查了视频超分的四个基本模块：`Propagation`, `Alignment`, `Aggregation`以及`Upsampling`。通过复用现有方案的模块并添加微小改动，作者提出了一种简单方案：BasicVSR，它在推理速度、复原质量方面取得了引人注目的提升。

 作者对BasicVSR进行了系统的分析，解释了性能提升的原因所在，同时也讨论了其局限性。在BasicVSR的基础上，作者进一步提出了“信息寄存(information-refile)”与“耦合传播(coupled propagation)”两种机制促进信息集成。所提BasicVSR及其改进IconVSR将视频超分的性能进行了更进一步的提升，可以作为视频超分领域的一个更强的基准。

 下图给出了所提方案与其他视频超分方案在性能与推理速度放慢的对比，可以看到BasicVSR与IconVSR遥遥领先。

![image-20210607150518854](/Users/neonrocks/Desktop/Super_Resolution_Typora/BasicVSR论文笔记.assets/image-20210607150518854.png)

#### Introduction

 作者对现有各式各样的VSR方案按照各个模块的功能(propagation, alignment, aggregation, upsampling)进行了拆分

- Propagation：在这里特指信息的流动，比如局部信息，单向信息流动，双向信息流动；
- Alignment：在这里特指对齐的类型以及有无；
- Aggregation：在这里指的是对齐特征的集成方式；
- Upsampling：在这里指的是上采样过程所采用的方案，基本都是Pixel-Shuffle。

 在上述四个功能单元中，Propagation和Alignment对性能和效率影响最大。双线传播有助于最大化的进行信息汇集，光流方案有助于进行相邻帧特征对齐。通过简单的上述模块组合所得的BasicVSR即取得了超越SOTA的指标与速度(指标提升0.61dB，推理速度快了24倍)。

 在BasicVSR的基础上，作者提出了如下两种新颖的扩展得到了IconVSR。

- 信息寄存，它采用了额外的模块提取从稀疏选择帧(比如关键帧)中提取特征，然后插入到主网络用于特征改善。
- 耦合传播，它促进了前向与反向传播分支中的信息交换。

 这两个模块不仅可以降低误差累积问题，同时可以获取更完整的时序信息以生成更高质量的特征，进而得到更好的重建结果。得益于上述两种设计，IconVSR以0.31dB指标提升超过了BasicVSR。

#### Method

##### Propagation

**Propagation** 是VSR中最具影响力的模块，它特指信息的利用方式。现有的传播机制可以分为一下三大类：

- Local Propagation: 滑动窗口的方法(比如RBPN，TGA，EDVR)采用局部窗口内的多帧LR图像作为输入并进行中间帧的重建。这种设计方式约束了信息范围，进而影响了模型的性能。下图给出了不同信息范围下的模型性能对比，可以看到：(1)**全局信息的利用具有更佳性能**;(2) **片段的两端性能差异非常大**，说明了长序列累积信息(即全局信息)的重要性。



![image-20210607151735660](/Users/neonrocks/Desktop/Super_Resolution_Typora/BasicVSR论文笔记.assets/image-20210607151735660.png)

- Unidirectional： 已有单向传播方案(比如RLSP、RSDN、RRN、FRVSR)采用了从第一帧到最后一帧的单向传播的方式，这种方式导致了不同帧接受的信息是不平衡的，比如第一帧只会从自身接受信息，而最后一帧则可以接受整个序列的信息。下图给出了单向传播与双向传播的性能差异对比。可以看到：(1)在早期，单向传播方案的PSNR指标严重低于双向传播方案；(2)整体来看，单向传播方案的指标要比双向传播的方案低0.5dB。

- Bidirectional：上述两种信息传播方案的弊端可以通过双向传播方案解决。BasicVSR采用了经典的双向传播机制，给定输入图像及其近邻帧，相应的特征传播分别描述为和，定义如下

  Given an LR image xi, its neighboring frames xi−1 and xi+1, and the corresponding features propagated from its neighbors, denoted as hfi−1 and hbi+1, we have

$h_{i}^{b}=F_{b}(x_{i}, x_{i+1},h_{i+1}^{b})$

$h_{i}^{f}=F_{f}(x_{i}, x_{i-1},h_{i-1}^{f})$

where $F_{b}$ and $F_{f}$ denote the backward and forward propagation branches, respectively.



##### Alignment

空间对齐在VSR中起着非常重要的作用，它负责将高度相关的的图像/特征进行对齐并送入到后续的集成模块。主流VSR方案可以分别以下三大类：

- Without Alignment: 现有的递归方案(比如RLSP、BRCN、RSDN、RRN)通常不进行对齐，非对齐的特征直接进行集成导致了次优的性能。作者在实验中发现：不添加对齐会导致1.19dB的指标下降，也就是说**对齐是非常有必要**。
- Image Alignment：早期的TOFlow采用光流在图像层面进行对齐，已有研究表明：相比图像层面对齐，**特征层面的对齐可以取得显著的性能提升**。
- Feature Alignment: 作者采用了类似TOFlow的光流方案，并用于特征对齐，对齐后的特征融入到后的残差模块中。这里采用的特征对齐可以描述如下：

其中分别表示光流估计、仿射变换以及残差模块。

##### Aggregation and Upsampling

 BasicVSR采用了非常基本的模块(残差模块以及PixelShuffle)用于特征集成与上采样，假设中间特征表示,这里的特征集成与上采样模块描述如下：

$s_{i}^{\{b,f\}}=S(x_{i},x_{i\pm1})$,

$\overline{h}_{i}^{\{b,f\}}=W(h_{i\pm1}^{\{b,f\}},s_{i}^{\{b,f\}})$,

$h_{i}^{\{b,f\}}=R_{\{b,f\}}(x_{i}, \overline{h}_{i}^{\{b,f\}})$

$y_{i}=U(h_{i}^{f},h_{i}^{b})$

 总而言之，BasicVSR采用了双向传播机制、特征层面的光流对齐、concate进行特征集成，pixelshuffle进行上采样。



#### IconVSR

 以BasicVSR作为骨干，作者引入了两种新颖的单元以消除传播过程中的误差累积促进时序信息集成。

- Information-Refil： 不精确的对齐会导致误差累积问题，尤其是该文所采用的长期传播方案。为消除上述问题，作者提出了信息寄存机制，见下图。



![image-20210607153817839](/Users/neonrocks/Desktop/Super_Resolution_Typora/BasicVSR论文笔记.assets/image-20210607153817839.png)

 它采用了额外的特征提取器提取关键帧与近邻帧的特征，所提取的特征将于对齐特征通过卷积进行融合。该过程描述如下：

其中分别表示特征提取器与卷积。融合后的特征将被融入到后续的残差模块中

- Coupled Propagation: 在双向传播中，特征经由相反的方向进行独立处理。作者对此添加了耦合传播机制，使得两者产生关联，见上图。

![image-20210607153921746](/Users/neonrocks/Desktop/Super_Resolution_Typora/BasicVSR论文笔记.assets/image-20210607153921746.png)

![image-20210607153940372](/Users/neonrocks/Desktop/Super_Resolution_Typora/BasicVSR论文笔记.assets/image-20210607153940372.png)

#### Conclusion

- **BasicVSR以全面优势超过了现有视频超分方案**，在UDM10数据集上，以0.61dB超过了RSDN且具有相当的参数量、更快的速度；
- **IconVSR可以进一步提升BasicVSR的指标高达0.31dB**。

## CSDN笔记

#### 主要贡献点

1. 重新分析了视频超分网络中的四大模块（网络传播、对齐、聚合和上采样）的作用，以及它们的优缺点。提出了一个基础的视频超分框架BasicVSR，并在Reds和Vimeo数据集上验证了该框架的有效性。
2. 扩展BasicVSR框架，设计了信息重新填充机制和成对传播策略，促进信息聚合，即IconVSR网络。



## OpenMMLab

新的`BasicVSR++` 已经出来了。





