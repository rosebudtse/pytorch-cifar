# 极耳图像质量评价
## 一，简介
### 1.1 分类介绍
对极耳图像进行分类，极耳图像的分类如下：
- 打光不均
- 抖动
- 过暗
- 过曝
- 模糊
- 尾部过大
- 棱镜脏污
- 正常
- 左超视野

其中8类异常类，1类正常类。

### 1.2 网络结构
使用EfficientNet中的EfficientNet-B3进行训练，在保证精度的同时压缩参数量。

论文地址：[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
官方源码：[Github](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)


EfficientNet的主干结构如下：
![主干](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9LWVNEVG1PVlp2cWljU3JzTEZpYVRSSWdJNUw0WjN1SjFFajdiS2YwZTR0VzQyMjNNZHBUbXFXRWhxdHdIenk2MGljMFQ4SGMwQUJ2cUo2eUFha3BPQVIzUS82NDA?x-oss-process=image/format,png)

每个主干包含7个block，这些block还有不同数量的子block,子block的数量随着EfficientNet-B0到EfficientNet-B7的增加而增加。

Efficient-B0中总层数为237层，而EfficientNet-B7总数为813层，为保证训练速度，我们选择EfficientNet-B3作为主干网络。所有的层都可以由下面的5个模块和上面的主干组成：
![模块](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9LWVNEVG1PVlp2cWljU3JzTEZpYVRSSWdJNUw0WjN1SjFFUTZrSG1iWUpFUHhqZ21lZWdqQlNrWmtlRWttSlRXWTlHd2pFaWN6SDFJblpsQzQ4RzJpYUFSZFEvNjQw?x-oss-process=image/format,png)

我们使用这5个模块来构建整个结构：
- **模块1**：这是子block的起点。
- **模块2**：此模块用于除第一个模块外的所有7个主要模块的第一个子block起点。
- **模块3**：它作为跳跃连接到所有的子block。
- **模块4**：用于将跳跃连接合并到第一个子block中。
- **模块5**：每个子block都以跳跃连接的方式连接到之前的子block中，并用此模块进行组合。

这些模块被进一步组合成子block，这些子block将在block中以某种方式使用。
![子block](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9LWVNEVG1PVlp2cWljU3JzTEZpYVRSSWdJNUw0WjN1SjFFZHFBUmRmdzBSOHUxTUVKUjdPSkxIdXhyaWNXaWM2YnhmOW5KS1Q1ODJTd0tyRmlhanFlVkRxRWliUS82NDA?x-oss-process=image/format,png)
- **子block1**：仅用作第一个block中的第一个子block。
- **子block2**：用作所有其他block中的第一个子block。
- **子block3**：用作所有block中除第一个子block（子block2）以外的任何子block。

以下是EfficientNet-B3的结构：
![EfficientNet-B3结构](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9LWVNEVG1PVlp2cWljU3JzTEZpYVRSSWdJNUw0WjN1SjFFVmNPSFFTQmlhbEdqVkw4SlppYnU0bmlidWJTTGt3a2VkQ3pJWEF0YU1LMGI5dVlINXk0NlgyZWZBLzY0MA?x-oss-process=image/format,png)

- EfficientNet-B3模型参数量为12M，相比于ResNet50的25M，参数量减少一半，同时精度也有所提升。
![参数量](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9LWVNEVG1PVlp2cWljU3JzTEZpYVRSSWdJNUw0WjN1SjFFdWliVGliSmVLMXpRNFJqWjE1c2ljMGljNjdMdWZLbDFzaWNCSVNIbWRhTGtqMUlQOGRSSGF6MzR5M1EvNjQw?x-oss-process=image/format,png)

- 网络结构中有Dropout层，避免特定的神经元过度拟合训练数据。Dropout 的基本思想是在网络的训练过程中，随机地将一些神经元的输出置为零，从而减少不同神经元之间的依赖关系，也就是“丢弃”一部分神经元。这样做的目的是使得模型不能过度依赖某些特定的神经元，从而提高模型的泛化能力。
- 
### 1.3 模型性能
对极耳数据集采用3个评价指标：**Top-1准确率**、**Top-3准确率**、**图像是否异常准确率**，说明详见**4.1 评价指标说明**。

**Top-1准确率**：
| 分类       | Top-1 Acc |
| ---------- | --------- |
| 打光不均   | 75.00%    |
| 抖动       | 70.97%    |
| 过暗       | 93.55%    |
| 过曝       | 87.88%    |
| 模糊       | 97.06%    |
| 尾部过大   | 58.62%    |
| 棱镜脏污   | 36.00%    |
| 正常       | 85.71%    |
| 左超视野   | 100.00%   |
| **总体**   | **80.43%** |

**Top-3 准确率**：
| 分类       | Top-3 Acc |
| ---------- | --------- |
| 打光不均   | 94.23%    |
| 抖动       | 80.65%    |
| 过暗       | 100.00%   |
| 过曝       | 100.00%   |
| 模糊       | 100.00%   |
| 尾部过大   | 100.00%   |
| 棱镜脏污   | 72.00%    |
| 正常       | 99.05%    |
| 左超视野   | 100.00%   |
| **总体**   | **95.38%** |

**异常识别成功率**：

| 分类       | 识别成功率 |
| ---------- | --------- |
| 打光不均   | 90.38%    |
| 抖动       | 83.87%    |
| 过暗       | 100.00%   |
| 过曝       | 96.97%    |
| 模糊       | 100.00%   |
| 尾部过大   | 58.62%    |
| 棱镜脏污   | 64.00%    |
| 正常       | 85.71%    |
| 左超视野   | 100.00%   |
| **总体**   | **86.60%** |

## 二，数据预处理
### 2.1 现有数据集
对极耳数据集进行分类，极耳数据集的结构如下：
```
The file structure is as following:
dataset
├──train
│   ├── daguang
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── doudong
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── ...
├──test
│   ├── daguang
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── doudong
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── ...
```
由于数据集各个分类不均衡，有分类图片过多或过少，因此数据集已使用imgaug经过预先增强。对于分类过少的分类，仅采用随机裁剪进行补齐式增强，来弥补与其他分类的数据在数量上的差距。在进行训练时会进行额外数据增强。

### 2.2 新增数据集
如果使用或合并新数据集进行继续训练，需执行`./augment`文件夹下的以下文件：
- `rename.py`：对数据集中的文件重新命名，执行增强的图片名称不能包含中文。
- `augment.py`：在`./augement`文件夹下新建aug文件夹，先将需要预先增强（补齐数量差）的类别放入其中，后执行文件，增强比例可以在代码中调节。
- `split_data.py`：将数据集按4: 1的比例切分为训练集和测试集，切分比例可以在代码中调节。

### 2.3 删除自动生成的文件夹
如果在jupyter notebook上训练，数据集中可能会自动生成隐藏的`./.ipynb_checkpoints`文件夹，在文件目录中并不可见，需在终端进行手动删除。
```
rm -rf ./.ipynb_checkpoints
```
**在代码执行过程中勤打印label，来避免出现`./.ipynb_checkpoints`文件夹，导致label不匹配报错的问题。**

## 三，模型训练
模型训练采用Pytorch框架，使用损失函数为交叉熵损失函数，默认学习率为0.001，训练时batch_size为64，训练轮数为200轮。训练时会自动保存最优模型，保存在`./output_new`文件夹下。

### 3.1 学习率优化器
优化器为**Adam优化器**，相较于SGD优化器，Adam优化器能够更快地收敛，且不容易陷入局部最优。
![优化器比较](https://upload-images.jianshu.io/upload_images/18628169-a47abfcdc80d5c7f.png?imageMogr2/auto-orient/strip|imageView2/2/w/589/format/webp)

Adam优化器相较于SGD优化器不容易卡在鞍点处，如下图所示：
![鞍点](https://pic1.zhimg.com/80/4a3b4a39ab8e5c556359147b882b4788_720w.webp)

### 3.2 学习率调度器
学习率调度器采用**余弦退火(Cosine Annealing)**，学习率通过以下公式更新：

$$lr = \frac{lr_{\text{max}}}{2} \left(\cos\left(\frac{\text{epoch} \cdot \pi}{\text{epochs}}\right) + 1\right)$$

代码如下：
```py
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```
学习率会在epoch从0-200之间以余弦函数的形式从0.001(初始学习率)递减到0。

一般来讲，$T-max \geq epoch-max$，即余弦退火的周期应该大于训练的总轮数，这样才能保证学习率在训练结束时趋于0。
### 3.3 图像增强
模型采用的增强方式为：
- 下采样至512*512
- 随机裁剪后添加padding
- 以随机概率添加随机高斯噪声

由于给定数据集极耳方向皆为从左到右，因此不采用随机水平翻转。
```py
transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomCrop(448, padding=32),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std_range=(0, 20), probability=0.2),
    transforms.Normalize((0.2117, 0.2117, 0.2117), (0.2194, 0.2194, 0.2194)),
])
```
### 3.4 开始训练

模型的分类数在`./models`文件夹下的`efficientnet.py`中进行修改，修改`num_classes`参数即可。

在终端执行以下命令即可开始训练：
```
python main.py
```
可以指定初始学习率开始训练：
```
python main.py --lr=0.01
```
从checkpoint继续训练，已经过一定轮次训练，**学习率已下降到较小数值**，因此一般指定更小的初始学习率继续训练：
```
python main.py -r --lr=0.0005
```
### 3.5 模型保存
模型默认每隔10个epoch进行一次评估，评估后保存当前轮次模型，保存在./output_new文件夹下。

同时会保存最优模型eff_best_model.pth，保存在./output_new文件夹下。

## 四，模型评估
执行以下命令即可对模型进行评估：
```
python test_all.py
```
### 4.1 评价指标说明
模型会输出三个指标：**Top-1准确率**、**Top-3准确率**、**异常识别成功率**。
- **Top-1准确率**：即在图片分类的所有可能中**概率最大**的分类。
- **Top-3准确率**：即在图片分类的所有可能中**概率前三**的分类。
- **异常识别成功率**：即**非正常的图像**识别为**非正常**；**正常的图像**识别为**正常**的准确率。

### 4.2 说明：采用Top-3分类的原因
数据集中存在图像属于某一类，但拥有多个分类的特征的情况存在。以**棱镜脏污**类为例：
- **异常**类识别为其他**异常**类：**打光不均**类中出现镜头脏污，识别为**棱镜脏污**
![打光不均识别为脏污](./images/daguang2zangwu.png)
- **异常**类识别为**正常**类：**棱镜脏污**类中并无镜头脏污，识别为**正常**
![脏污识别为正常](./images/zangwu2zhengchang.png)
- **正常**类识别为**异常**类：**正常**类中出现镜头脏污，识别为**棱镜脏污**
![正常识别为脏污](./images/zhengchang2zangwu.png)

因此，为了解决存在的多类别问题和类别中存在包含关系的问题，对各个分类进行Top-3准确率计算，对其他可能的两个预测进行一定程度的宽容。

## 五，模型推理
执行以下命令即可对指定图像进行推理：
```
python inference.py --img_path train_test/test/zhengchang/147.png
```
脚本将输出指定图像的Top-3的数值、类别索引以及各自的置信度，如下图所示：
![推理输出](./images/inference.png)

## 六，已知问题
1，由于数据集较小，且数据集不同分类之间较不均衡，模型有过拟合的倾向。由于EfficientNet中有Dropout，过拟合有一定程度的缓解，可以尝试采用更高层数的EfficientNet模型，如EfficientNet-B7。

2，在数据增强时，是用了随机裁剪＋padding的增强方法，会导致这使得“尾部过大”分类训练结果较差。因为“尾部过大”分类的特征是极耳总长度过长，裁剪后图像中极耳的部分不完整，无法区分是否过长，即无法区分“尾部过大”与“正常”。如果取消随机裁剪+padding，容易导致过拟合。

3，数据集中有分类错误的图片和含有除自己分类特征外其他特征的图片，如**4.2 说明：采用Top-3分类的原因**中所述，进行数据清洗可以提高准确率，但是新的数据集中仍然存在这种情况，进行数据清洗后训练模型反而会导致模型在新数据集上表现不佳，因此采用Top-3分类的方法可以一定程度上缓解这种情况。

## 七，优化思路
### 7.1 模型优化
#### 7.1.1 优化器
在模型训练前期，可以使用Adam优化器进行快速收敛，在训练后期使用SGD优化器进行微调。
- **数据更稳定**：在模型训练的后期阶段，数据往往更加稳定，梯度信息较为可靠。SGD在此时可能表现得更好，因为它具有较小的学习率，可以在目标函数的最小值处跳动并逐渐收敛
- **局部最优解**： Adam在初期对于探索解空间和快速收敛非常有效，但在后期可能会导致过多的探索，从而使模型陷入局部最优解或不能达到更好的全局最优解。SGD可能会更加保守，帮助模型在后期更好地搜索和收敛。
#### 7.1.2 数据读取
类比增强学习的思想，在训练的每轮Epoch中，都随机抽样训练集的80%进行训练，从而引入更多的随机性。这有助于模型在训练过程中更好地探索不同的样本组合，并提高模型的泛化能力。另外，使用随机样本的抽样，还可以降低模型对于特定批次或训练样本顺序的敏感性，避免过拟合的现象。

### 7.2 后处理优化
在**4.2 说明：采用Top-3分类的原因**中提到，图片可能包含多个分类的特征而导致分类不准确，但仍有方法可以迎难而上，提高模型的准确率。

> 以**棱镜脏污**类为例，已知棱镜脏污容易被识别为**正常**类：

如果能找到某一个条件，使得满足该条件时，被误判为**正常**类的图片可以被修正为**棱镜脏污**，同时尽量不影响原本被正确识别的**正常**类图片，不让这些正确的图片反而被误判为**棱镜脏污**，那么就可以提高**棱镜脏污**的准确率。

对groundtruth为**棱镜脏污**的图片来说：
$$H0: \text{识别为脏污} \quad \quad H1: \text{识别为正常}$$
我们要减小犯第一类错误的概率，即减小拒绝H0而接受H1的情况，同时要尽可能减小犯第二类错误的概率，即减小把**正常**的图片误判为**棱镜脏污**的情况。

那么一个基本的算法是对所有groundtruth为**棱镜脏污**类却被识别为**正常**类的图片的分类的置信度进行判断，如果
$$\left\{
\begin{array}{l}
f(\text{正常}) < g(P(\beta))\\[10pt]
f(\text{脏污}) > g(P(\alpha))
\end{array}
\right.$$
其中$f(\text{正常})$是图片被识别为**正常**类的置信度，$f(\text{脏污})$是图片被识别为**脏污**类的置信度，$g(P(\beta))$是有关犯第二类错误（即把**正常**的图片误判为**棱镜脏污**）的一个函数，$g(P(\alpha))$是有关犯第一类错误（即把**棱镜脏污**的图片误判为**正常**）的一个函数，若满足这个条件，即把错判的**正常**修正为**棱镜脏污**。

$g(P(\alpha))$和$g(P(\beta))$需要对**正常**和**棱镜脏污**中的图片的置信度进行统计，分别找到两个函数能尽量拟合它们的分布。以下是一个简单的例子，取$g(P(\beta)) = 0.7$，$g(P(\alpha)) = 0.24$，**棱镜脏污**类的准确率提高了12%，同时对**正常**类的准确率影响不大。
![脏污结果](./images/zangwu.png)
![正常结果](./images/zhengchang.png)

这只是取了两个极其简单的常数，提升尚且如此，如果函数能更好地拟合分布，提升的准确率会更高。

而且这仅考虑了一个二分类问题，如果能考虑其他分类的影响，则**棱镜脏污**类的准确率会更高。如果应用到其他分类上，理论上来说在n维空间上可以找到这么一个判断条件的函数，尽量去拟合，数据集的准确率会提高很多。
