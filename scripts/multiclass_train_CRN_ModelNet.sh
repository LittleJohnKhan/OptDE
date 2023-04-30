VIRTUALDATA=CRN
REALDATA=ModelNet
VCLASS=plane,car,chair,lamp,couch,table
RCLASS=plane,car,chair,lamp,sofa,table
LOGDIR=logs
CUDA_VISIBLE_DEVICES=$1 python multiclass_trainer.py \
--virtualdataset ${VIRTUALDATA} \
--realdataset ${REALDATA} \
--num_class 6 \
--visual_class_choices ${VCLASS} \
--real_class_choices ${RCLASS} \
--split train \
--epoch 200 \
--mask_type k_mask \
--save_inversion_path ./${LOGDIR}/${VIRTUALDATA}_${REALDATA}_multiclass \
--dataset_path /home/zhaojiacheng/Dataset/unpaired_pcl_completion/virtual-scan/CRN/ \
--ckpt_load pretrained_models/ \
--log_dir ${LOGDIR}

# add pretrained discriminator for Frechet Inception Distance?