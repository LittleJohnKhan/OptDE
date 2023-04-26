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
