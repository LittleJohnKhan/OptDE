
# todo List
- [x] use the latest verison of torch and torchvision instead of the older version claimed in requirements.txt
- [x] revise shell script for CRN path
- [ ] read optde.py
- [ ] need to learn k-mask from shapeinversion
- [ ] vscode 自定义代码折叠区域?
- [ ] 学习任务： pcd编程共性的内容，如数据集处理代码，metrics计算代码，Encoder/Decoder Backbone以及其权重提取等等内容
- BUG win10 vscode source control 配置, git能够使用但在vscode中不能够

# reading task
- [ ] model/gcn.py
- [ ] model/network.py
- [ ] utils/
- [ ] evaluation/ ????
- [ ] external/ CD 距离使用方法，EMD距离在shapeinversion中
- [x] data/ datasets processing

| Class     | CRN | 3D_FUTURE | ModelNet40 | KITTI | Scannet| Matterport |
--------------------------------------------------------------------------
| plane      |     |          |     y      |       |        |            |
| cabinet    |     |  y       |            |       |        |            |
| car        |     |          |     y      |       |        |            |
| chair      |     |  y       |     y      |       |        |            |
| lamp       |     |  y       |     y      |       |        |            |
| couch(sofa)|     |  y       |     y      |       |        |            |
| table      |     |  y       |     y      |       |        |            |
| boat       |     |          |            |       |            |
# Project structure
- win10 -> github // `git push/fetch`
- win10 -> zjc-2080ti // `sftp`
zjc-2080ti only runs the code, no code editing

# Training Procedure

train.py
```python
class Trainer:
    def __init__(): # 4 dataloaders
        self.virtual_train_dataloader
        self.real_train_dataloader
        self.virtual_train_dataloader
        self.real_train_dataloader

    def train():

    '''important parameters'''
        bool_virtual_train = True       
        bool_real_train                 # only after 160 epoches
        bool_domain_train = True        # real training set 4 losses
        bool_cons_train                 # only after 120 epoches cs_loss switch permutation consistency
        bool_virtual_test = True        # compute and log on virtual test set
        bool_real_test = True           # compute and log on real test set

        train_cd_loss_mean -> [train_cd_loss,] # reconstruction and completion
        train_ucd_loss_mean ->[train_ucd_loss,] # reconstruction and completion
        train_di_loss_mean -> [di_loss,] # domain classification, domain-invariant
        train_ds_loss_mean -> [ds_loss,] # domain classification, domain-specific
        train_vp_loss_mean -> [vp_loss,] # view-point MSE loss
        train_cs_loss_mean -> [cs_loss,] # factor permutation consistency loss 

        # TODO optimization stage loss ?

    '''training procedure'''
        for epoch in range(200):
            for iter in range(144):
                1. train on virtual scan*2, call train_virtual_one_batch()  # cd
                    input (self.partial, self.gt)
                    |-  ftr_loss : MSE between complete shape x and gt
                    |-  rec_loss : cd between x_rec and partial
                    |-  com_loss : cd between x and gt
                2. train on real scan,      call train_real_one_batch()     # ucd
                    input (self.partial)
                    |-  rec_loss :  cd partial <=> x_rec
                    |-  com_loss :  ucd partial => x
                    |-  mask_loss x_map=preprocess(x) ucd x_map => partial
                3. train domain loss,       call train_domain_one_batch()   # 4 loss

            1. test on virtual scan,        call test_one_batch()
            2. test on real scan            call test_real_one_batch

```

optde.py
```python
class OptDE:
    def __init__():
        # network modules
        self.models['Encoder']  = self.Encoder = pointnet   #(B,N,3)->(B,1024)
        self.models['Decoder']  = self.Decoder = treeGAN    #(B,96x3)->(B,2048,3)
        self.models['DI']       = self.DI_Disentangler  = Disentangler(f_dims=96)  #(B,1024)->(B,96), fs
        self.models['MS']       = self.MS_Disentangler  = Disentangler(f_dims=96)  #(B,1024)->(B,96), fo
        self.models['DS']       = self.DS_Disentangler  = Disentangler(f_dims=96)  #(B,1024)->(B,96), fd
        self.models['DIC']      = self.DI_Classifier    = Classifier(f_dims=96)    #(B,96)->(B,2)
        self.models['DSC']      = self.DS_Classifier    = Classifier(f_dims=96)    #(B,96)->(B,2) 
        self.models['VP']       = self.V_Predictor      = ViewPredictor(f_dims=96) #(B,96)->(B,2) (\rho,\theta)
        self.models['D']        = self.D=Discriminator(features=args.D_FEAT) #(B,N,3)->(B,1) pretrained shapeinversion discriminator

        # loss functions
        self.ftr_net = self.D
        self.criterion = DiscriminatorLoss(self.ftr_net,..)  
        self.di_criterion = nn.CrossEntropyLoss() # virtual_label=0, real_label=1
        self.ds_criterion = nn.CrossEntropyLoss() # virtual_label=0, real_label=1
        self.vp_criterion = nn.MSELoss()
        self.consistency_criterion = nn.MSELoss()
        self.directed_hausdorff = DirectedHausdorff()


    # supporting functions
    def train_virtual_one_batch(curr_step, ith=-1, complete_train=False):
        return train_cd_loss
    def train_real_one_batch(curr_step, epoch, ith=-1):
        return train_ucd_loss
    def train_domain_one_batch(curr_step, alpha, switch_idx_default=None, ith=-1):
        return  di_loss, ds_loss, vp_loss, cons_feature
    def train_consistency_one_batch(curr_step, cons_feature, return_generated=False, ith=-1):
        return cs_loss

```

model/gcn.py
```python
class TreeGCN
TODO in  main.py

```

model/network.py
```python 

class Discriminator                   Param # pretrained from shapeinversion
=================================================================
Discriminator                            --
├─ModuleList: 1-1                        --
│    └─Conv1d: 2-1                       256
│    └─Conv1d: 2-2                       8,320
│    └─Conv1d: 2-3                       33,024
│    └─Conv1d: 2-4                       65,792
│    └─Conv1d: 2-5                       131,584
├─LeakyReLU: 1-2                         --
├─Sequential: 1-3                        --
│    └─Linear: 2-6                       65,664
│    └─LeakyReLU: 2-7                    --
│    └─Linear: 2-8                       8,256
│    └─LeakyReLU: 2-9                    --
│    └─Linear: 2-10                      65
│    └─Sigmoid: 2-11                     --


class Generator                         Param #
=================================================================
Generator                                --
├─Sequential: 1-1                        --
│    └─TreeGCN: 2-1                      83,200
│    │    └─ModuleList: 3-1              73,728
│    │    └─Sequential: 3-2              1,566,720
│    │    └─LeakyReLU: 3-3               --
│    └─TreeGCN: 2-2                      131,584
│    │    └─ModuleList: 3-4              139,264
│    │    └─Sequential: 3-5              1,310,720
│    │    └─LeakyReLU: 3-6               --
│    └─TreeGCN: 2-3                      262,656
│    │    └─ModuleList: 3-7              204,800
│    │    └─Sequential: 3-8              1,310,720
│    │    └─LeakyReLU: 3-9               --
│    └─TreeGCN: 2-4                      524,544
│    │    └─ModuleList: 3-10             135,168
│    │    └─Sequential: 3-11             983,040
│    │    └─LeakyReLU: 3-12              --
│    └─TreeGCN: 2-5                      262,400
│    │    └─ModuleList: 3-13             151,552
│    │    └─Sequential: 3-14             327,680
│    │    └─LeakyReLU: 3-15              --
│    └─TreeGCN: 2-6                      524,544
│    │    └─ModuleList: 3-16             167,936
│    │    └─Sequential: 3-17             327,680
│    │    └─LeakyReLU: 3-18              --
│    └─TreeGCN: 2-7                      33,554,432
│    │    └─ModuleList: 3-19             4,320
│    │    └─Sequential: 3-20             167,680
│    │    └─LeakyReLU: 3-21              --



class Disentangler                     Output Shape              Param #
==========================================================================================
Disentangler                             [10, 96]                  --
├─Linear: 1-1                            [10, 1024]                1,049,600
├─BatchNorm1d: 1-2                       [10, 1024]                2,048
├─Linear: 1-3                            [10, 96]                  98,400
├─BatchNorm1d: 1-4                       [10, 96]                  192

class Classifier                       Output Shape              Param #
==========================================================================================
Classifier                               [10, 2]                   --
├─Linear: 1-1                            [10, 24]                  2,328
├─BatchNorm1d: 1-2                       [10, 24]                  48
├─Linear: 1-3                            [10, 2]                   50

class ViewPredictor                    Output Shape              Param #
==========================================================================================
ViewPredictor                            [10, 2]                   --
├─Linear: 1-1                            [10, 96]                  9,312
├─BatchNorm1d: 1-2                       [10, 96]                  192
├─Linear: 1-3                            [10, 2]                   194


class PCN                              Output Shape              Param #
==========================================================================================
Encoder                                  [10, 1024]                --
├─Conv1d: 1-1                            [10, 128, 2048]           512
├─BatchNorm1d: 1-2                       [10, 128, 2048]           256
├─Conv1d: 1-3                            [10, 256, 2048]           33,024
├─BatchNorm1d: 1-4                       [10, 256, 2048]           512
├─Conv1d: 1-5                            [10, 512, 2048]           262,656
├─BatchNorm1d: 1-6                       [10, 512, 2048]           1,024
├─Conv1d: 1-7                            [10, 1024, 2048]          525,312
├─BatchNorm1d: 1-8                       [10, 1024, 2048]          2,048

Decoder                                  [10, 2048, 3]             --
├─Linear: 1-1                            [10, 1024]                1,049,600
├─BatchNorm1d: 1-2                       [10, 1024]                2,048
├─Linear: 1-3                            [10, 1024]                1,049,600
├─BatchNorm1d: 1-4                       [10, 1024]                2,048
├─Linear: 1-5                            [10, 3072]                3,148,800
├─Conv1d: 1-6                            [10, 512, 2048]           527,360
├─BatchNorm1d: 1-7                       [10, 512, 2048]           1,024
├─Conv1d: 1-8                            [10, 512, 2048]           262,656
├─BatchNorm1d: 1-9                       [10, 512, 2048]           1,024
├─Conv1d: 1-10                           [10, 3, 2048]             1,539


```


# 知识点整理
## Gradient reverse
implementation from https://github.com/fungtion/DANN_py3/blob/master/functions.py
                    https://blog.csdn.net/weixin_38132729/article/details/122594503
```python
from torch.autograd import Function
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
# usage
virtual_f_di = ReverseLayerF.apply(virtual_f_di, alpha)
real_f_di = ReverseLayerF.apply(real_f_di, alpha)
```


## Some language details
`torch.Tensor.item()`为张量自带函数，适用于单个张量将张量转为Python数值
`torch.Tensor.tolist()`适用于多个张量元素转换
```python
a=torch.Tensor([1.0])   # torch.Tensor
b=a.item()              # float
c=torch.Tensor([1.0,2.0,3.0]) # torch.Tensor
d=c.tolist()            # list
```

## nn.CrossEntropyLoss
```python
import torch.nn as nn
import torch
# create data x and label y
x = torch.tensor([[0.7459, 0.5881, 0.4795], #(B,3)
                [0.2894, 0.0568, 0.3439],
                [0.6124, 0.7558, 0.4308]])
y = torch.tensor([0,1,1]) #(B,)
# softmax
softmax = nn.Softmax(dim=1)
x_softmax = softmax(x)
# log
x_log = torch.log(x_softmax)
# negative log likelihood
loss_func = nn.NLLLoss()
loss_func(x_log, y)
>>>tensor(1.0646)

loss_func = nn.CrossEntropyLoss()
loss_func(x,y)
>>>tensor(1.0646)
```
## Visualization
matplotlib
```python
#-*-coding:utf-8-*-
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from ipdb import set_trace
 
def readXYZfile(filename, Separator):
  data = np.loadtxt(filename,delimiter = Separator)[:, 0:3]
  point = [data[:,0].tolist(),data[:,1].tolist(),data[:,2].tolist()]
  return point
 
#三维离散点图显示点云
def displayPoint(data,title):
  #散点图参数设置
  fig=plt.figure() 
  ax=Axes3D(fig) 
  ax.set_title(title) 
  ax.scatter3D(data[0], data[2], data[1], c = 'r', marker = '.') 
  ax.set_xlabel('x') 
  ax.set_ylabel('y') 
  ax.set_zlabel('z') 
  plt.show()
 
if __name__ == "__main__":
  data = readXYZfile("airplane_0003.txt",',')
  displayPoint(data, "airplane_0003")
```

open3d
```python
#-*-coding:utf-8-*-
import numpy as np 
import open3d as o3d
 
points = np.loadtxt('airplane_0003.txt', delimiter=',')[:, 0:3]
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
# 设置点云显示的颜色（0-1）
point_cloud.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([point_cloud])
# o3d.visualization.draw([point_cloud])
# 保存点云到ply
# o3d.io.write_point_cloud("test.ply", point_cloud)

```

## FOO BAR
foo/bar是自二战时的俚语FUBAR(Fucked Up Beyond All Repair)，就是坏到无法修缮的意思。国外的程序员用这些词很大程度上是为了幽默。这些词没有任何意义，通常被当做占位符来使用，可以代表任何东西。
当变量，函数，或命令本身不太重要的时候， foobar , foo , bar ,baz 和 qux 就被用来充当这些实体的名字，这样做的目的仅仅是阐述一个概念，说明一个想法。这些术语本身相对于使用的场景来说没有任何意义。

## PyTorch summary打印网络结构
```bash
pip install torchinfo
conda install -c conda-forge torchinfo
```
```python
model = simpleNet()
batch_size = 64
summary(model, input_size=(batch_size, 3, 32, 32))
```

## jump between folders
```bash
(base) ➜  ~ tldr z
z
Tracks the most used (by frecency) directories and enables quickly navigating to them using string patterns or regular expressions.More information: https://github.com/rupa/z.

 - Go to a directory that contains "foo" in the name:
   z {{foo}}

 - Go to a directory that contains "foo" and then "bar":
   z {{foo}} {{bar}}

 - Go to the highest-ranked directory matching "foo":
   z -r {{foo}}

 - Go to the most recently accessed directory matching "foo":
   z -t {{foo}}

 - List all directories in z's database matching "foo":
   z -l {{foo}}

 - Remove the current directory from z's database:
   z -x .

 - Restrict matches to subdirectories of the current directory:
   z -c {{foo}}

```

## yeild
TODO https://blog.csdn.net/takedachia/article/details/123931246