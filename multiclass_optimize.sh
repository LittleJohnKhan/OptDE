VIRTUALDATA=CRN
REALDATA=3D_FUTURE
VCLASS=cabinet,chair,lamp,couch,table
RCLASS=cabinet,chair,lamp,couch,table
LOGDATE=Log_2022-10-10_13-57-27         #? 手动指定
LOGDIR=logs
CUDA_VISIBLE_DEVICES=$1 python multiclass_optimizer.py \
--virtualdataset ${VIRTUALDATA} \
--realdataset ${REALDATA} \
--num_class 5 \
--visual_class_choices ${VCLASS} \
--real_class_choices ${RCLASS} \
--split train \
--epoch 150 \
--mask_type k_mask \
--save_inversion_path ./${LOGDIR}/${VIRTUALDATA}_${REALDATA}_multiclass_finetune \
# --ckpt_load pretrained_models/${VCLASS}.pt \    #! load ckpt
--finetune_ckpt_load ./${LOGDIR}/${VIRTUALDATA}_${REALDATA}_multiclass/${LOGDATE}/multiclass.pt \ #! load ckpt
--dataset_path /home/zhaojiacheng/Dataset/unpaired_pcl_completion/virtual-scan/CRN/ \
--log_dir ${LOGDIR}
