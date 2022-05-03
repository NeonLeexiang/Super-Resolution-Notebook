# Super Resolution notebook

图像超分辨总结

* SRCNN
* VDSR
* SRGAN
* EDSR
* EDVR
* BasicVSR

---

## SRCNN

**图像超分辨经典论文**

### 理论

**不同的先验知识，会指向不同的结果。我们的任务，就是学习这些先验知识。**目前效果最好的办法都是基于样本的（example-based）

* _稀疏编码_ 知识补充 稀疏编码 `sparse coding`

  **字典学习**

  （1）它实质上是对于庞大数据集的一种降维表示，或者说是信息的压缩（2）正如同字是句子最质朴的特征一样，字典学习总是**尝试学习蕴藏在样本背后最质朴的特征**（假如样本最质朴的特征就是样本最好的特征）。

  **稀疏表示的本质：用尽可能少的资源表示尽可能多的知识，这种表示还能带来一个附加的好处，即计算速度快。**

  

**论文提出一种有趣的视角：CNN 所构造的模型和稀疏编码方法（sparse coding based）是等价的。**稀疏编码方法的流程如下： 

1. 从原始图片中切割出一个个小块，并进行预处理（归一化）。这种切割是密集的，也就是块与块之间有重叠；

2. 使用低维词典（low-resolution dictionary）编码，得到一个稀疏参数；

3. 使用高维词典（high-resolution dictionary）结合稀疏参数进行重建（换了个密码本）；

4. 将多个小块拼接起来，重合部分使用加权和拼接。

![image-20210709140646161](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210709140646161.png)

上图是卷积神经网络对应于稀疏编码的结构。对于一个低分辨率图像 Y，第一个卷积层提取 feature maps。第二个卷积层将 feature maps 进行非线性变换，变换为高分辨率图像的表示。最后一层恢复出高分辨率图像。 

相比于稀疏编码，论文提出的模型是 end-to-end 的，便于优化。并且，不需要求最小二乘的解，运算速度更快。

* _端到端_

  端到端模型仅使用一个模型、一个目标函数，就规避了前面的多模块固有的缺陷，这是它的优点之一；另一个优势是减少了工程的复杂度，一个网络解决所有步骤，也就是「炼丹」。

### 模型信息

* 输入输出

![image-20210709141721450](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210709141721450.png)



| **模型内容** | 参数 |
| ------------ | ---- |
| 损失函数     | MSE  |
| 评判标准     | PSNR |

* 模型

![image-20210709142550350](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210709142550350.png)

![image-20210709143026862](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210709143026862.png)



* 模型效果

![image-20210709144141275](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210709144141275.png)

![image-20210709144224903](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210709144224903.png)



### 代码 model

```python
class pytorch_SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(pytorch_SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(9, 9), padding=(9//2, 9//2))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=(5, 5), padding=(5//2, 5//2))
        """
        inplace=True
        对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量。
        """
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
```



### 论文重点

**SRCNN 网络是 CNN 应用在超分辨率重建领域的开山之作。**虽然论文尝试了更深的网络，但是相比于后来的神经网络，如 DRCN 中的网络，算是很小的模型了。受限于模型的表达能力，最终训练的结果还有很大的提升空间。 

另外，虽然相比于 sparse coding 方法，SRCNN 可以算是 end to end 方法了。但是仍然需要将图片进行 bicubic 差值到同样大小。此后的 ESPCN 使用 sub-pixel convolutional layer，减少了卷积的运算量，大大提高了超分辨率重建的速度。 

在复现的过程中，笔者发现 SGD 收敛速度相当慢，论文中曲线横轴都是数量级。使用 Adam 优化器，收敛速度更快，并且几个模型的 PSNR 值更高。说明使用 SGD 训练时候，很容易陷入局部最优了。







---

## VDSR



本文针对SRCNN，提出了3个问题的解决方案：

1. 卷积核尺寸一定时，网络层数过浅导致生成图片的感受野过小。一个更深度的网络势必能带来更大的感受野，这就使得网络能够利用更多的上下文信息，能够有更反映全局的映射；

2. 为解决问题1必将加深网络深度，会使得网络的收敛速率变慢甚至无法收敛（学习率、梯度裁减）。作者提出的解决方案是利用残差学习和更高的学习率。首先，我们认为LR和HR共享了很多基本信息（由后边可知：低频结构信息），所以我们只需要学习LR和HR之间的差值（高频信息）即可，这称之为残差学习。明显可知，传统的学习方法（SRCNN）是从LR中直接学习到完整的结果图（SR）的特征，并且中间处理过程需要完整的保留输入信息，这学习强度明显大于只学习到部分高频信息（即残差学习），所以残差学习无论从难以程度还是学习时间成本上都优于SRCNN。最后再将学到的高频信息和LR（低频信息）整合即可获得接近于目标HR的结果图SR。其次，利用更高的学习率可以加速网络的收敛（事实上，作者对于不同的epoch运用了不同的学习率），但可能造成梯度的爆炸，所以提出了梯度裁减的方法来避免。可以看出，后来的网络大多都运用了残差学习的思想；

3. SRCNN不能进行多尺度放大。分别设计不同的网络生成不同的模型来解决不同的尺度放大问题不切合实际，所以作者提出用一个网络训练不同的尺度放大图片来得到一个模型，解决不同放大尺度的问题。（由后文可知，作者并未给出具体操作，或许和EDSR相同，**待考证**。）

### Related work

**感受野**

感受野定义为：输出图像中每个像素能够反映输入图像区域的大小。一个更深度的网络势必能带来更大的感受野，这就使得网络能够利用更多的上下文信息，能够有更全局的映射。

其次，残差学习通过对LR图片学习高频细节，然后加到LR上以获得SR图像。这可以更好的理解很多作者讲的**ill-posed problem**，因为我们是从低频图LR上估计高频图。另外我们可以联想到retinex的思想，也是将一副图片进行拆分。

```markdown
适定问题(well-posed problem)和不适定问题(ill-posed problem)都是数学领域的术语。

前者需满足三个条件，若有一个不满足则称为"ill-posed problem"：

1. a solution exists     

 解必须存在

2. the solution is unique       

解必须唯一

3. the solution's behavior changes continuously with the initial conditions. 

解能根据初始条件连续变化，不会发生跳变，即解必须稳定

```

```markdown
在计算机视觉中，有很多任务不满足“适定”条件，通常不满足第二条和第三条。

比如用GAN“伪造”图像的时候，这个任务就不满足“解的唯一性”。

做图像超分辨率，或者对图像去雨去雾去模糊等等任务时，这些都没有一个标准答案，解有无数种。更重要的是，这些解都是不稳定的。

Jaeyoung在CVPR的论文中这样描述CV中的不适定问题：

In most cases, there are several possible output images corresponding to a given input image and the problem can be seen as a task of selecting the most proper one from all the possible outputs.

这种不适定问题就是：一个输入图像会对应多个合理输出图像，而这个问题可以看作是从多个输出中选出最合适的那一个。 

```

### 模型结构 and 代码

![image-20210712140540442](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210712140540442.png)

**代码**

```python
class ConvReLUBlock(nn.Module):
    def __init__(self):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                              stride=(1, 1), padding=(1, 1), bias=False)
        """
            inplace=True
            对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量。
        """
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class pytorch_VDSR(nn.Module):
    def __init__(self, num_channels=1):
        super(pytorch_VDSR, self).__init__()
        self.residual_layer = self.make_layer(ConvReLUBlock, 18)
        self.input_layer = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3, 3,),
                                     stride=(1, 1), padding=(1, 1), bias=False)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3),
                                      stride=(1, 1), padding=(1, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        out = self.relu(self.input_layer(x))
        out = self.residual_layer(out)
        out = self.output_layer(out)
        out = torch.add(out, residual)
        return out

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


```



### Conclusion

1. VDSR的一个重要缺点是在进入网络之前做图像的插值放大，这使得网络参数增大计算量增加。

2. 这个网络可能能用在去噪、压缩图像去伪影等图像重建方向，值得尝试。



---

## SRGAN

### 前置内容

之前超分的研究虽然主要聚焦于“恢复细粒度的纹理细节”这个问题上，但将问题一直固定在**最大化峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)**上，等价于 最小化与GT图像的均方重建误差(mean squared reconstruction error, MSE)。

​												$PSNR = 10 \times log_{10}(\frac{(2^n-1)^2}{MSE}), n=8$

而这也就导致：

1. **高频细节(high-frequency details)** 的丢失，整体图像过于平滑/模糊；
2. 与人的视觉**感知不一致**，超分图像的精确性与人的期望不匹配（人可能更关注前景，而对背景清晰度要求不高）。

从而提出3个改进：

1. 新的backbone：SRResNet；
2. GAN-based network 及 新的损失函数：
3. adversarial loss：提升**真实感**(photo-realistic natural images)；
4. content loss：获取HR image和生成图像的**感知相似性(perceptual similarity)**，而不只是像素级相似性(pixel similarity)；或者说特征空间的相似性而不是像素空间的相似性。
5. 使用主观评估手段：MOS，更加强调人的感知。

_TODO：content loss，MOS，loss=adversarial loss_

![image-20210712142319504](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210712142319504.png)

![image-20210712142330468](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210712142330468.png)



### SRGAN 网络结构 和 代码复现

![image-20210712142436423](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210712142436423.png)

```python
class ResidualBlock(Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = BatchNorm2d(channels)
        self.p_relu = PReLU()
        self.conv2 = Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.p_relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return residual + x


class UpSampleBlock(Module):
    def __init__(self, in_channels, up_scale):
        super(UpSampleBlock, self).__init__()
        self.conv = Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=(3, 3), padding=(1, 1))
        self.pixel_shuffle = PixelShuffle(up_scale)
        self.p_relu = PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.p_relu(x)
        return x


class Generator(Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = Sequential(
            Conv2d(3, 64, kernel_size=(9, 9), padding=(4, 4)),
            PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = Sequential(
            Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(64)
        )

        # up sampling
        block8 = [UpSampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(Conv2d(64, 3, kernel_size=(9, 9), padding=(4, 4)))
        self.block8 = Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        return (torch.tanh(block8) + 1) / 2


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1)),
            LeakyReLU(0.2),

            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64),
            LeakyReLU(0.2),

            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(128),
            LeakyReLU(0.2),

            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128),
            LeakyReLU(0.2),

            Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(256),
            LeakyReLU(0.2),

            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256),
            LeakyReLU(0.2),

            Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(512),
            LeakyReLU(0.2),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            LeakyReLU(0.2),

            AdaptiveAvgPool2d(1),
            Conv2d(512, 1024, kernel_size=(1, 1)),
            LeakyReLU(0.2),
            Conv2d(1024, 1, kernel_size=(1, 1)),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

```

### SRGAN实验结果及分析

消融实验说明：

1. skip-connection结构的有效性；
2. PSNR体现不出人的感知（MOS）；
3. GAN-based Network能更好捕捉一些人的感知细节（高频信息？），MOS更高；
4. VGG特征重建也有助于捕捉图像的部分感知细节。

_TODO：VGG 还有 MOS_



---

## EDSR

- 做出的修改主要是在残差网络上。残差结构的提出是为了解决high-level问题，而不能直接套用到超分辨这种low-level视觉问题上。因此作者移除了残差结构中一些不必要的模块，结果证明这样确实有效果。
- 另外，作者还设置了一种多尺度模型，不同的尺度下有绝大部分参数都是共用的。这样的模型在处理每一个单尺度超分辨下都能有很好的效果。

### Contribution

Since batch normalization layers normalize the features, they get rid of range flexibility from networks by normalizing the features。因此有必要把batch norm层移除掉。另外，和SRResnet相似，相加后不经过relu层。最终的结构图如下：

![image-20210712143802410](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210712143802410.png)

- 值得注意的是，bn层的计算量和一个卷积层几乎持平，移除bn层后训练时可以节约大概40%的空间。
- 太多的残差块会导致训练不稳定，因此作者采取了residual scaling的方法，即残差块在相加前，经过卷积处理的一路乘以一个小数（比如作者用了0.1）。这样可以保证训练更加稳定。

### Model and  code

![image-20210712143944570](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210712143944570.png)



```python
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False


'''
    for bias:
        CLASS torch.nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
                              
        padding='valid' is the same as no padding. 
        padding='same' pads the input so the output has the shape as the input. 
        However, this mode doesn’t support any stride values other than 1.
        
        
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        
        
'''


'''
    according to the paper, they use res_scale=0.1 to set the res_block because:
    
    `EDSR` 的残差缩放 `Residual Scaling`    
  
  
    EDSR 的作者认为提高网络模型性能的最简单方法是增加参数数量，堆叠的方式是在卷积神经网络中，堆叠多个层或通过增加滤波器的数量。
    当考虑有限的复合资源时，增加宽度 (特征Channels的数量) F 而不是深度(层数) B 来最大化模型容量。
    但是特征图的数量增加(太多的残差块)到一定水平以上会使训练过程在数值上不稳定。
    残差缩放 (residual scaling) 即残差块在相加前，经过卷积处理的一路乘以一个小数 (作者用了0.1)。
    在每个残差块中，在最后的卷积层之后放置恒定的缩放层。
    当使用大量滤波器时，这些模块极大地稳定了训练过程。
    在测试阶段，该层可以集成到之前的卷积层中，以提高计算效率。
    使用上面三种网络对比图中提出的残差块（即结构类似于 SRResNet ，但模型在残差块之外没有 ReLU** 层）构建单尺度模型 EDSR。
    此外，因为每个卷积层仅使用 64 个特征图，所以单尺度模型没有残差缩放层。
'''


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, padding, bias=False, bn=False, act=nn.ReLU(inplace=True), res_scale=0.1):
        super(ResBlock, self).__init__()
        m = []

        for i in range(2):
            m.append(nn.Conv2d(in_channels=n_feats, out_channels=n_feats,
                               kernel_size=kernel_size, padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=False):
        m = []
        '''
            &是按位逻辑运算符，比如5 & 6，5和6转换为二进制是101和110，此时101 & 110=100，100转换为十进制是4，所以5 & 6=4
            
            
            如果一个数是2^n，说明这个二进制里面只有一个1。除了1.
            a  = (10000)b
            a-1 = (01111)b
            a&(a-1) = 0。
            如果一个数不是2^n，
            说明它的二进制里含有多一个1。            
            a = (1xxx100)b            
            a-1=(1xxx011)b         
            那么 a&(a-1)就是 (1xxx000)b，            
            而不会为0。
            
            所以可以用这种方法判断一个数是不2^n。
            
        '''

        '''
            一：与运算符（&）
            运算规则：
            0&0=0；0&1=0；1&0=0；1&1=1           
            即：两个同时为1，结果为1，否则为0
        '''

        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats,
                                   kernel_size=(3, 3), padding=(1, 1), bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(in_channels=n_feats, out_channels=9 * n_feats,
                               kernel_size=(3, 3), padding=(1, 1), bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, n_channels=3, n_resblocks=32, n_feats=256, scale=4, res_scale=0.1, rgb_range=1):
        super(EDSR, self).__init__()

        self.n_channels = n_channels
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.scale = scale
        self.res_scale = res_scale
        self.rgb_range = rgb_range

        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.act = nn.ReLU(True)

        self.sub_mean = MeanShift(self.rgb_range)
        self.add_mean = MeanShift(self.rgb_range, sign=1)

        net_head = [nn.Conv2d(self.n_channels, self.n_feats, kernel_size=self.kernel_size, padding=self.padding)]

        net_body = [
            ResBlock(
                n_feats=self.n_feats, kernel_size=self.kernel_size, padding=self.padding,
                act=self.act, res_scale=self.res_scale
            ) for _ in range(self.n_resblocks)
        ]

        net_body.append(nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_feats,
                                  kernel_size=self.kernel_size, padding=self.padding))

        net_tail = [
            Upsampler(self.scale, self.n_feats, act=False),
            nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_channels,
                      kernel_size=self.kernel_size, padding=self.padding)
        ]

        self.net_head = nn.Sequential(*net_head)
        self.net_body = nn.Sequential(*net_body)
        self.net_tail = nn.Sequential(*net_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.net_head(x)

        res = self.net_body(x)
        res = torch.add(x, res)

        x = self.net_tail(res)
        x = self.add_mean(x)

        return x

```

### More details:

作者提出的模型主要是提高了图像超分辨的效果，并赢得了NTIRE2017超分辨率重建挑战赛。

做出的修改主要是在ResNet上。

作者移除了残差结构中一些不必要的模块如BN层，结果证明这样确实有效果。

另外，作者还提出了一种多尺度模型，不同的尺度下有绝大部分参数都是共享的。

这样的模型在处理每一个单尺度超分辨下都能有很好的效果。



### BN层的意义

#### BN介绍

Batch Norm可谓深度学习中非常重要的技术，不仅可以使训练更深的网络变容易，加速收敛，还有一定正则化的效果，可以防止模型过拟合。在很多基于CNN的**分类任务**中，被大量使用。

但在**图像超分辨率**和**图像生成**方面，BatchNorm的表现并不是很好。当这些任务中，网络加入了BatchNorm层，反而使得训练速度缓慢且不稳定，甚至最后结果发散。

#### 超分中BN层不好的原因

以图像超分辨率来说，网络输出的图像在色彩、对比度、亮度上要求和输入一致，改变的仅仅是分辨率和一些细节。

而Batch Norm，对图像来说类似于一种对比度的拉伸，任何图像经过Batch Norm后，其色彩的分布都会被归一化。

也就是说，它破坏了图像原本的对比度信息，所以Batch Norm的加入反而影响了网络输出的质量。

ResNet可以用BN，但也仅仅是在**残差块**当中使用。

还是回到**SRResNet**，上图的(b)就是一个用于图像超分辨率的残差网络。

#### SRResNet使用BN层的原因

ResNet中引入了一种叫残差网络结构，其和普通的CNN的区别在于**从输入源直接向输出源多连接了一条传递线**来*恒等映射*，用来进行残差计算。

可以把这种连接方式叫做identity shortcut connection,或者我们也可以称其为skip connection。

其效果是为了防止网络层数增加而导致的梯度弥散问题与退化问题。



### 分类问题的BN的好处

图像分类不需要保留图像的对比度信息，利用图像的结构信息就可以完成分类。

所以，将图像信息都通过BatchNorm进行归一化，反而降低训练难度。

甚至，一些不明显的结构，在BatchNorm后也会被凸显出来（对比度被拉开）。



### 风格迁移的BN的好处

风格化后的图像，其色彩、对比度、亮度均和原图像无关，而只与风格图像有关。

原图像只有结构信息被表现到了最后生成的图像中。



### 简而言之

BN会是网络训练时使数据包含忽略图像像素（或者特征）之间的绝对差异（因为均值归零，方差归一），而只存在相对差异。

所以在不需要绝对差异的任务中（比如分类），BN提升效果。

而对于图像超分辨率这种需要利用绝对差异的任务，BN会适得其反。



### 残差缩放(residual scaling)的意义

EDSR的作者认为提高网络模型性能的最简单方法是增加参数数量，堆叠的方式是在卷积神经网络中，堆叠多个层或通过增加滤波器的数量。

当考虑有限的复合资源时，**增加宽度(特征Channels的数量)F**而不是深度(层数)B来最大化模型容量。

但是特征图的数量增加(太多的残差块)到一定水平以上会使训练过程在数值上不稳定。

残差缩放(residual scaling)即残差块在相加前，经过卷积处理的一路乘以一个小数(作者用了0.1)。

在每个残差块中，在最后的卷积层之后放置恒定的缩放层。

当使用大量滤波器时，这些模块极大地稳定了训练过程。

在测试阶段，该层可以集成到之前的卷积层中，以提高计算效率。

使用上面三种网络对比图中提出的残差块（即结构类似于SRResNet，但模型在残差块之外没有ReLU层）构建单尺度模型EDSR。

此外，因为每个卷积层仅使用64个特征图，所以单尺度模型没有残差缩放层。



### 损失函数使用L1而不是L2的原因

训练时，损失函数用L1而不是L2，即根据LapSRN的思想采用了L1范数来计算对应的误差，**L2损失会导致模糊的预测**。

BN有一定的正则化效果，可以不去理会Dropout，L2正则项参数的选择。

除此之外，更深层的原因是是实际图像可能含有多种特征，对应有关的图像构成的真实分布。

图像特征分布有许多个峰值，比如特征1是一个峰，特征2是一个峰...

对于这种图像分布，我们称之为：**多模态(Multimodal)**。

假如用MSE（或者L2）作为损失函数，其潜在的假设是我们采集到的样本是都来在同一个高斯分布。

但是生活中的实际图像具有多种特征，而且大部分图像分布都不只有一个峰。

如果强行用一个单峰的高斯分布，去拟合一个多模态的数据分布，例如两个峰值。

因为损失函数需要减小生成分布和数据集经验分布（双峰分布）直接的差距，而生成分布具有两种类型，模型会尽力去“满足”这两个子分布，最后得到的优化结果。







---

# 视频超分辨

## EDVR

作者认为要解决视频增强，必须要解决两大问题：

1. 图像对齐（Alignment）。

视频相邻帧存在一定的抖动，必须先对齐才能进一步处理融合。以往这可以使用光流算法处理，但本文中作者发明了一种新的网络模块PCD 对齐模块，使用Deformable卷积进行视频的对齐，整个过程可以端到端训练。

2. 时空信息融合（Fusion）。

挖掘时域（视频前后帧）和空域（同一帧内部）的信息融合。本文中作者发明了一种时空注意力模型进行信息融合。

![image-20210714103015560](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210714103015560.png)

其中PCD 对齐模块，使用金字塔结构级联的Deformable卷积构建，如下图：

![image-20210714103046162](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210714103046162.png)

作者发明的时空注意力融合模型TSA如下图：

![image-20210714103102289](/Users/neonrocks/Desktop/Super_Resolution_Typora/SuperResolution_notebook.assets/image-20210714103102289.png)



代码：所有代码均可参考BasicVSR

