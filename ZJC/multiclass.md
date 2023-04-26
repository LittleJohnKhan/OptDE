# 3种可行的多分类方法
1. domain，class显式分开，用shapenet pretrained的分类网络在partial上进行微调,  作为类别先验器。
    - 该先验器抽取到的class特征如何使用？ (拼接/数学加减/…
        
2. domain-class信息作为一个整体factor，对抗学习思想，将判别器从2分类换成2K分类，交叉熵损失约束
    - domain-class factor经过判别器应输出one-hot encoding
    - shape factor 经过判别器应输出均匀分布，即没法区分来自于哪个domain,class
        
3. domain-class信息作为一个整体factor，对比学习思想
    - domain-class factor 使用SwAV方式训练不需要直接跟负样本比，K个class则人为设定2K个聚类中心，同类同域所有数据对比均为正样本。$z$代表当前样本的domain-class feature, $C$代表domain-class prototype训练过程中更新，Code为cluster assignment。C如何初始化？
    - shape factor 该咋学习？


# TODO-TREE
1. 实现multiclass baseline
2. 分支branch 
    AdversarialLearning
    ContrastiveLearning
    PartialClassPrior
    * master
    remotes/origin/HEAD -> origin/master
    remotes/origin/master

3. 改多分类对抗 OptDE.train_domain_one_batch() ->
    原文中 DI_Classifier 与 DS_Classifier 使用相同结构的网络与损失，只不过DS_Classifier前还用了梯度反转
    [ ]折腾vscode本地编写之后同步文件夹到linux上，git更新用win10端完成
    [ ]改为多类分别之后，DS_classifier有两种训练方法，其一还是梯度反转，哪个更加好？ classifier 重写
    [x] 修改两个shell脚本
    [x] 重写train.py/trainer_optimizer.py，dataloader勿动。训练过程中每一个epoch仅训练一个class，每个class将会被训练150轮，每轮中训练145iter
    [x] multiclass_trainer.py main函数编写 测试 real_data_loader是否为空
   

【 】

  

# Notes
- source:CRN class= ['plane','cabinet','car','chair','lamp','couch','table','watercraft'], target:3D_FUTURE, class=  ['cabinet',     ,'chair','lamp','sofa','table']
- 假定source和target包含相同的class，dataloader每个batch读入同domain同class的数据


['MatterPort', 'ScanNet', 'KITTI'] 三个数据集直接指定包含数据集-class的路径；
['CRN', '3D_FUTURE'] 两个数据集指定数据集路径，class由shell脚本指定

run.sh

run_optimizer.py
```bash
VIRTUALDATA=CRN
REALDATA=3D_FUTURE
VCLASS=plane
RCLASS=plane
LOGDATE=Log_2022-07-14_14-43-57 # need to change
LOGDIR=logs
CUDA_VISIBLE_DEVICES=$1 python trainer_optimizer.py \
--virtualdataset ${VIRTUALDATA} \
--realdataset ${REALDATA} \
--class_choice ${VCLASS} \
--split train \
--epoch 200 \
--mask_type k_mask \
--save_inversion_path ./${LOGDIR}/${REALDATA}_finetune_${RCLASS} \
--ckpt_load pretrained_models/${VCLASS}.pt \
--finetune_ckpt_load ./${LOGDIR}/${REALDATA}_${RCLASS}/${LOGDATE}/${VCLASS}.pt \
--dataset_path /home/zhaojiacheng/Dataset/unpaired_pcl_completion/virtual-scan/CRN/ \
--log_dir ${LOGDIR}
```