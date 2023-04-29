VIRTUALDATA=CRN
REALDATA=3D_FUTURE
VCLASS=cabinet,chair,lamp,couch,table
RCLASS=cabinet,chair,lamp,sofa,table
LOGDATE=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX         #? 手动指定
LOGDIR=logs
CUDA_VISIBLE_DEVICES=$1 python multiclass_optimizer.py \
--virtualdataset ${VIRTUALDATA} \
--realdataset ${REALDATA} \
--num_class 5 \
--visual_class_choices ${VCLASS} \
--real_class_choices ${RCLASS} \
--split train \
--epoch 200 \
--mask_type k_mask \
--save_inversion_path ./${LOGDIR}/${VIRTUALDATA}_${REALDATA}_multiclass_finetune \
--ckpt_load pretrained_models/ \
--finetune_ckpt_load ./${LOGDIR}/${VIRTUALDATA}_${REALDATA}_multiclass/${LOGDATE}/multiclass.pt \
--dataset_path /home/zhaojiacheng/Dataset/unpaired_pcl_completion/virtual-scan/CRN/ \
--log_dir ${LOGDIR}
