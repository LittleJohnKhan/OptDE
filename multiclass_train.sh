VIRTUALDATA=CRN
REALDATA=3D_FUTURE
VCLASS=cabinet,chair,lamp,couch,table
RCLASS=cabinet,chair,lamp,couch,table
LOGDIR=logs
CUDA_VISIBLE_DEVICES=$1 python multiclass_trainer.py \
--virtualdataset ${VIRTUALDATA} \
--realdataset ${REALDATA} \
--num_class 5 \
--visual_class_choices ${VCLASS} \
--real_class_choices ${RCLASS} \
--split train \
--epoch 150 \
--mask_type k_mask \
--save_inversion_path ./${LOGDIR}/${VIRTUALDATA}_${REALDATA}_multiclass \
--dataset_path /home/zhaojiacheng/Dataset/unpaired_pcl_completion/virtual-scan/CRN/ \
--log_dir ${LOGDIR}

# TODO add pretrained model
# --ckpt_load pretrained_models/${VCLASS}.pt \    #! 不包含预训练权重

