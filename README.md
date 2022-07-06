记录了自己在学习深度学习时的各类问题，以便自己参考
_总结了一部分自己在学习过程中，遇到的频繁的点，重点难点问题！_
# 一、模型结构加深理解
## 1. BatchNorm2d
[原理解释](https://blog.csdn.net/weixin_44278406/article/details/105554268)
[基础 | batchnorm原理及代码详解](https://blog.csdn.net/qq_25737169/article/details/79048516)
==注意==：BatchNorma2d() 在激活函数之前

_BatchNorm什么时候**bias**设置为True？什么时候设置为False?_
 bn后有relu，bn的bias为False； bn后无relu，bn的bias为True

_relu的**inplace**什么时候设置为True，什么时候为False？_
[https://blog.csdn.net/qiumokucao/article/details/121115800](https://blog.csdn.net/qiumokucao/article/details/121115800)

###  BN & SyncBN 
[(BN 与 多卡同步 BN 详解](https://zhuanlan.zhihu.com/p/337732517)
个人理解：
BN的效果受到Batch_size的影响
而对于一个占用内存较高的model，batch_size往往设计的比较小，甚至为1，那么BN的效果大打折扣，采用全局的平均值与方差进行归一化，即SyncBN

## 2. with torch.no_grad() 
[with torch.no_grad() 详解](https://blog.csdn.net/weixin_46559271/article/details/105658654)

## 3. 反卷积，上采样，卷积
### 区别
[https://www.cnblogs.com/abella/p/10304654.html](https://www.cnblogs.com/abella/p/10304654.html)
[反卷积(Deconvolution)、上采样(UNSampling)与上池化(UnPooling)](https://blog.csdn.net/A_a_ron/article/details/79181108)
### 反卷积
反卷积容易造成不均匀的重叠，尤其是卷积核的大小不能被步幅整除时：
[https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)
[https://buptldy.github.io/2016/10/29/2016-10-29-deconv/](https://buptldy.github.io/2016/10/29/2016-10-29-deconv/)
[https://zhuanlan.zhihu.com/p/363564590](https://zhuanlan.zhihu.com/p/363564590)
 
## 4.dropout——待查询
[https://blog.csdn.net/junbaba_/article/details/105673998](https://blog.csdn.net/junbaba_/article/details/105673998)

## 5. 深度可分离卷积![在这里插入图片描述](https://img-blog.csdnimg.cn/b090205d09d64e3b9d8e4113dc3bf5e5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LiA6Z2i5ZCs5LiA6Z2i6Zeu,size_20,color_FFFFFF,t_70,g_se,x_16)
[https://zhuanlan.zhihu.com/p/92134485](https://zhuanlan.zhihu.com/p/92134485)
[https://blog.csdn.net/qq_33869371/article/details/103588818](https://blog.csdn.net/qq_33869371/article/details/103588818)
通    常   卷   积：
比如Image：(7,7,3)   conv是(3,3,3)   output就变成5*5*1

深度可分离卷积：
Image(7,7,3)   conv是(3,3,3)    得到  (5,5,3)
(5,5,3) 再经过conv(1,1,3)的卷积，得到结果 (5,5,1)

## 6. self.children() 与 self.modules()
假设网络结构为：


        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), 
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
                nn.Linear(n_hidden_1, n_hidden_2),
                nn.ReLU(True))
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

**self.children()输出顺序:** 
	每一个layer作为一个子块，输出是layer0,layer1.layer2，（只输出第一代儿子模块）
	
**self.modules() 输出顺序为**：
(1)self.layer1——(2)layer1的儿子nn.Linear——(3)layer1的儿子nn.ReLU——(4)self.layer2——(5)self.layer2的儿子nn.Linear——(6)self.layer2的儿子nn.ReLU——(7)self.layer3

通俗的说：self.modules()会用深度遍历优先的方式遍历每一个子模块
[self.modules() 和 self.children()](https://zhuanlan.zhihu.com/p/238230258)
补充：[Pytorch中named_children()和named_modules()的区别](https://blog.csdn.net/watermelon1123/article/details/98036360)

[更多关于 named_children,   named_parameters(),   named_modules(),参考官网](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
named函数和其他函数区别就是输出同时，多输出一个名称
for name,i in named_chidlren():   print(name);print(i); 

## 7. nn.AdaptiveAvgPool2d((a,b))
将任意一张2维数据size为(n,m)  
n,m可以比a，b大，也可以比a,b小
函数自动设置 kernel_size,stride,padding等等参数
使得无论输入size是多少的数据，输出size都是 (a,b)

stride = floor (nput_size / output_size）
kernel_size = input_size − (output_size−1) * stride

## 8. bottleneck——为了减少参数量
[residual bottleneck](https://blog.csdn.net/u011304078/article/details/80683985)

## 9 模型结构调整
### 9.1 模型容器—— nn.ModuleList 与 nn.Sequential
[https://zhuanlan.zhihu.com/p/75206669](https://zhuanlan.zhihu.com/p/75206669)
### 9.2 PyTorch-网络的创建，预训练模型的加载 
[PyTorch-网络的创建，预训练模型的加载](https://www.cnblogs.com/wangguchangqing/p/11058525.html) 
[\[Pytorch进阶技巧(一)\] 使用add_module替换部分模型](https://blog.csdn.net/qq_31964037/article/details/105416291)
## 10. hook到底有什么用？
[pytorch中的钩子（Hook）有何作用？](https://www.zhihu.com/question/61044004/answer/183682138)
[Pytorch中autograd以及hook函数详解](https://oldpan.me/archives/pytorch-autograd-hook)
自己的理解：
前向计算与后向传播过程中，对于中间变量，pytorch的运行机制为，输出结果，中间变量释放。 然而有时我们**希望获得中间变量，进行额外操作**，比如观察模型每一层在关注什么，输出特征图等等，此时就需要用到hook，

如火车从起点开到终点，而我希望火车到**一些站点**时，给我一些我想哟要的信息，于是在这些站点做了**标记**，这些让模型提供反馈的站点标记就是——hook

## 10、view(),transpose(),narrow(),expend()与contiguous()
==特别注意一个坑：==
view()，trainspose()，narow()和 expend()都会改变数据本身，只有contiguous不会
前三个类似于浅拷贝，，加入contiguous后等于深拷贝，
[https://blog.csdn.net/kdongyi/article/details/108180250](https://blog.csdn.net/kdongyi/article/details/108180250)
（获取数据地址：data_ptr()
### contiguous
[contiguous
](https://zhuanlan.zhihu.com/p/64551412)

# 二、训练技巧（涨点技巧）
## 1. 动态迭代率：
[pytorch官网解释](https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate)
[https://www.cnblogs.com/wanghui-garcia/p/10895397.html](https://www.cnblogs.com/wanghui-garcia/p/10895397.html)
## 2. warm up 
非常好用的涨点技巧，常常结合cosine函数
[base model第七弹：warm up、consine衰减 、标签平滑、apex、梯度累加](https://blog.csdn.net/zgcr654321/article/details/106765238?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.compare&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.compare)
正确公式参考：[https://blog.csdn.net/qq_40268672/article/details/121145630](https://blog.csdn.net/qq_40268672/article/details/121145630)


## 3. miou
[https://blog.csdn.net/u012370185/article/details/94409933](https://blog.csdn.net/u012370185/article/details/94409933)

## 4. 权重初始化
待查询————————————
[https://www.cnblogs.com/tangjunjun/p/13731276.html](https://www.cnblogs.com/tangjunjun/p/13731276.html)
[pytorch系列 -- 9 pytorch nn.init 中实现的初始化函数 uniform, normal, const, Xavier, He initialization](https://blog.csdn.net/dss_dssssd/article/details/83959474)

### 初始化随机数种子的设置
torch.manual_seed() 与 torch.cuda.manual_seed() 与 random.seed()


## 5. 图像增强技巧
[Pytorch：transforms的二十二个方法](https://blog.csdn.net/weixin_38533896/article/details/86028509)

[pytorch图像数据增强7大技巧](https://zhuanlan.zhihu.com/p/91477545)

[**如何对image与label做同样的增强处理？**](https://www.pythonf.cn/read/130189)

**[涨点技巧——mixup](https://zhuanlan.zhihu.com/p/380501504)**


## 6.model.train()和model.eval()
主要影响bn 和relu的权值
model.train() 时，会改变二者权值
model.eval()时，不改变二者权值
[https://blog.csdn.net/qq_38410428/article/details/101102075](https://blog.csdn.net/qq_38410428/article/details/101102075)

## 7. 涨点技巧——使用自动混合精度(AMP)
参考：[https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/](https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/)
**模型提速2~5倍左右，亲测有效**
[显存的原理与分析](https://mp.weixin.qq.com/s/wIWYY1t_vkwxf3z-69fn-g)
[https://oldpan.me/archives/how-to-calculate-gpu-memory](https://oldpan.me/archives/how-to-calculate-gpu-memory)
[如何在Pytorch中精细化利用显存](https://oldpan.me/archives/how-to-use-memory-pytorch)
[再次浅谈Pytorch中的显存利用问题( 附完善显存跟踪代码)](https://oldpan.me/archives/pytorch-gpu-memory-usage-track)

```python
import torch
# 在训练开始时创建一次
scaler = torch.cuda.amp.GradScaler()
for data, label in data_iter:
   optimizer.zero_grad()
   # 将运算转换为混合精度
   with torch.cuda.amp.autocast():
      loss = model(data)
   # 缩放损失，并调用backward()创建缩放梯度
   scaler.scale(loss).backward()
   # 取消缩放梯度和调用或跳过optimizer.step()
   scaler.step(optimizer)
   # 为下一次迭代更新比例
   scaler.update()
```
## 8. 涨点技巧——开启 cudNN benchmarking
假如，我们所训练的模型架构保持固定且输入大小保持不变，则可以尝试去设置 
torch.backends.cudnn.benchmark = True  
这将会启用 cudNN 自动调谐器，它使得 cudNN 能够测试多种不同的卷积计算方法，然后选择并使用其中最快的方法。
在测试案例中，Conv2d 卷积的前向传播中实现了 70% 的加速，在相同的前向和后向传播中实现了 27% 的加速卷积运算。



[pyTorch 进行更快的深度学习训练-指南](https://mp.weixin.qq.com/s?__biz=Mzg2MjU1MzU4Mw==&mid=2247487873&idx=1&sn=647a9756de29aaf4f36617f3fca682b6&chksm=ce074501f970cc17a846464d70edf669531b5e97a69182d5571d536b22692e7ab2565f481d03&scene=183&subscene=90#rd)
## 9. 涨点技巧——label_smooth
[原理](https://zhuanlan.zhihu.com/p/343807710)

## 10.  梯度累积
当显卡内存不足以支持大的batch_size,采用梯度累计的方法进行近似操作，
[显存不够用怎么办 —— 梯度累积](https://featurize.cn/notebooks/0b77a5ff-5f3f-4061-add3-1710bb6541cf)
缺点：BN层的批次大小还是原来的数值，没有相应的累积

# 其他

 - 1.[全连接层——理解&为什么不具有空间结构特征？](https://blog.csdn.net/weixin_45127897/article/details/124034805)（理解4）
 - 2.参数量params与计算次数FLOPs的计算 	
 [如何计算模型的参数量与浮点运算速度](https://blog.csdn.net/qq_40507857/article/details/118764782)
      	[CNN 模型所需的计算力flops是什么？怎么计算？](https://zhuanlan.zhihu.com/p/137719986)
     
   	补充：[getModelSize()](https://blog.csdn.net/qq_43219379/article/details/124003959)
      				获取 Tensor 的元素个数 ，a.numel() 等价 a.nelement()
     理解：参数量计算是衡量模型效率的评价指标；计算次数FLOPs（floating point oprations) 通俗理解：发生了多少次参数的计算，如加减乘除等。二者都是评价模型复杂程度的指标。
     更进一步，**参数的存在需要消耗内存，反应了空间复杂度；计算次数越高，消耗时间越多，反应了时间复杂度**
     注意：同一模型，不同的输入尺寸，具有不同的计算次数

# 三、保存与读取
模型的checkpoint的保存与读取
dataloader 的相关辅助函数
## 1. os.walk , os.listdir
###  os.walk()
读取文件某一目录下的文件
[Python中os.walk()的使用方法](https://zhuanlan.zhihu.com/p/149824829)
os.walk一般是形如：
```python
for curdir,dirs,files in os.walk(path):
```
curdir——返回当前读取路径
dirs    —— 当前路径下的文件夹
files   —— 返回当前路径下的文件（文件夹不返回）

walk顾名思义，会逐级的走下去，直到走到最后一层，
例如：
第一次循环：  返回path的curdir,dirs,files情况
第二次循环：返回path中第一个文件夹的 curdir,dirs,files 情况，
以此类推
### os.listdir()
不想要分文件文件夹这么麻烦了，直接返回给我路径下所有的文件名吧
**os.listdir()**——一步到位

## glob
待补充
[glob](https://blog.csdn.net/GeorgeAI/article/details/81035422)


## 2. 模型checkpoint保存——torch.save
（*为了方便加载和调用，在模型训练中间某一阶段对模型状态进行保存——一般我们称之为checkpoint*）
torch.save有两种方式，
torch.save(model,path) ——保存整个模型
torch.save(modle.state_dict , path) ——  保存模型的参数

加载时 **torch.load(path)**

用第一种方法看起来方便，但容易出错，尤其在cuda和cpu的对应上。
参考：
[https://www.cnblogs.com/kk17/p/10074188.html](https://www.cnblogs.com/kk17/p/10074188.html)
[checkpoint——想进入模型训练的某一个阶段](https://www.cnblogs.com/qinduanyinghua/p/9311410.html)
[https://blog.csdn.net/qq_40520596/article/details/106955452](https://blog.csdn.net/qq_40520596/article/details/106955452)

## 3. state_dict()
非常常用的两个函数：
torch.save(model.state_dict)    
model.load_state_dict()

有时候想修改模型部分结构，但是还是想要预训练的参数，如何读取预训练的部分结构训练值：
[https://blog.csdn.net/Jee_King/article/details/86423274](https://blog.csdn.net/Jee_King/article/details/86423274)
[https://www.jb51.net/article/178720.htm](https://www.jb51.net/article/178720.htm)
## 路径问题小技巧
 1. rstrip(char）
 	例如Str_A.rstrip('.jpg')，就去除了字符Stri_A 末尾的.jpg，默认去除空格
 
 2. os.sep
 	不同的操作系统，文件路径分隔符不同，（ Linux——/	Window——\ ）如何让系统自动选择合适的分隔符呢？
 	os.sep——就是自动选择的合适的分隔符
 	[os.sep用法](https://blog.csdn.net/qq_18483627/article/details/105365191)
 
 3. os.path.basename
 	返回一段路径的最后一个文件名
 	如:root = D:/Desktop/pic/abc , 	则返回abc
 	注意:root如果以/ 或者\ 结尾，输出为空
 	
 4.  os.mkdir() 与 os.makedirs()  与 os.path.exist()
 	  [os.mkdir() 与 os.makedirs()的区别](https://blog.csdn.net/gqixf/article/details/80180640)
 	  os.exist() 输出某路径是否存在
 
 5. os.path.isfile() 与 os.path.isdir()
 	os.path.isfile(  ) 判断括号内的对象 是不是一个文件
 	os.path.isdir( ) 判断括号内的对象，是不是目录
 6. os.listdir()
 	列出括号的路径下的所有文件和文件夹
 	（不会打开每个文件夹查看是否还有文件，只会返回指定路径下的文件）
 	
 7. split 与 rsplit
 	split()：能把文件按照括号内指定符号进行分割，默认空格，如
	```python
	hh = 'D:/Desktop/jena'
	kk = hh.split('/')
	```
	就能得到：	![在这里插入图片描述](https://img-blog.csdnimg.cn/f921f1bfa3384d878093a181cf32a084.png)
也可以指定分割次数：
 如  a.split(‘、’，1） 就从头开始，分割1次，得到两个
有时希望**从末尾开始分割**怎么办？
用  rstrip()
	```python
	hh = 'D:/Desktop/jena/jena_000001_000019_leftImg8bit_foggy_beta_0.005.png'
	aa = hh.rsplit('/',1)
	print(aa)
	```
		
 

# 四、绘图，结果输出
## 3.1评价指标
[https://blog.csdn.net/sinat_29047129/article/details/103642140](https://blog.csdn.net/sinat_29047129/article/details/103642140)
## 3.2 混淆矩阵
[https://www.cnblogs.com/qi-yuan-008/p/11675499.html](https://www.cnblogs.com/qi-yuan-008/p/11675499.html)
## 3.3 visdom 模块
[pytorch调用visdom模块](https://www.codeleading.com/article/97292674745/)
步骤

 1. window + R：cmd   打开cmd窗口，
 2. conda env list——查看都有什么工作路径
 3. activate 其中一个路径（不需要的忽略步骤2，3）
 4. 输入：python -m visdom.server
	 	就打开了一个窗口，回跳出一个链接，浏览器输入链接，就到了visdom的空界面![在这里插入图片描述](https://img-blog.csdnimg.cn/f6c491dc7a624e17b4c87ff788f02818.png)
 5. 运行想运行的代码，就可以在这里看到图像（注意窗口要一致，窗口编号看CMD的地方）

HTML语言：如 td、tl 、th等等，
[https://blog.csdn.net/qq_32067151/article/details/80013344](https://blog.csdn.net/qq_32067151/article/details/80013344)

补充：[https://zhuanlan.zhihu.com/p/34692106](https://zhuanlan.zhihu.com/p/34692106)
[https://www.zhihu.com/search?type=content&q=visdom](https://www.zhihu.com/search?type=content&q=visdom)



# 五、常见报错反馈
### 1、error--CUDA error: device-side assert triggered
问题描述&解决：
深度学习图像分割任务中，观察蒙版图(mask)数组，发现是[0,255]的RGB数组，而正确的应该是类别数组
通过数组的引用，将RGB数组转化为类别数组

### 2、RuntimeError: CUDA out of memory.
解决：[RuntimeError: CUDA out of memory.](https://blog.csdn.net/weixin_43760844/article/details/113462431)
[消除PyTorch的CUDA内存溢出报错](https://mp.weixin.qq.com/s/Cch_kb6mXKYcg4s-1vaDSw)
### 3、RuntimeError: expected scalar type Byte but found Float
数据类型不一致，
解决：[RuntimeError: expected scalar type Byte but found Float](https://blog.csdn.net/tang330023555/article/details/118733184)

# 附录一、安装问题
## anaconda、pycharm、pytorch
[Anaconda+Pycharm环境下的PyTorch配置方法](https://blog.csdn.net/aa3615058/article/details/89339790)
[https://blog.csdn.net/weixin_45127897/article/details/118914488](https://blog.csdn.net/weixin_45127897/article/details/118914488)

# 附录二、pytorch文档
官方文档往往有极高的质量，非常值得一看
[pytorch官方文档](https://pytorch.apachecn.org/#/README)
[pytorch中文网](https://www.pytorchtutorial.com/category/tutorial-pytorch-note/)
# 附录三、数据集：
[https://blog.csdn.net/weixin_45127897/article/details/124034805](https://blog.csdn.net/weixin_45127897/article/details/124034805)
## 总结与补充
常见语义分割数据集总结![](https://img-blog.csdnimg.cn/8cfa727d09084098b98dd0cab661b3de.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5LiA6Z2i5ZCs5LiA6Z2i6Zeu,size_20,color_FFFFFF,t_70,g_se,x_16)大规模图像分割数据集还包括 Semantic Boundaries Dataset(SBD)、 Microsoft Common Objects in COntext (COCO)、 KITTI、Adobe’s Portrait Segmentation、Youtube-Objects、Materials IN Context (MINC)、Densely-Annotated VIdeo Segmentation (DAVIS)、Stanford background、SiftFlow 以及 3D 数据集, 包括 ShapeNet Part、Stanford 2D-3D-S、A Benchmark for 3D Mesh Segmentation、Sydney Urban Objects Dataset、Large-Scale Point Cloud Classification Benchmark 等.   
 # 附录四、发展历史——建议的学习顺序

![在这里插入图片描述](https://img-blog.csdnimg.cn/3e378c2fcc1e4622b56d3affde96ce7e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5LiA6Z2i5ZCs5LiA6Z2i6Zeu,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/a5ec2c5d3ea44a06842cccd99d5e41c8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5LiA6Z2i5ZCs5LiA6Z2i6Zeu,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/fed22c540368421c86e7252558225916.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LiA6Z2i5ZCs5LiA6Z2i6Zeu,size_17,color_FFFFFF,t_70,g_se,x_16)


### 参考文献
[1]全卷积神经网络图像语义分割方法综述
[2]基于全卷积网络的图像语义分割方法综述
[3]基于深度学习的图像语义分割技术综述
[4]基于深度神经网络的图像语义分割研究综述
[5]……………………


