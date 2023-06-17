mkdir /project/train/dataset
cp -r /home/data/* /project/train/dataset

rm /project/train/dataset/images/*

python /project/train/src_repo/ev_detection/voc_label.py
python /project/train/src_repo/ev_detection/split_train_val.py



CUDA_VISIBLE_DEVICES=0 
python /project/train/src_repo/ev_detection/trainv8.py
# python /project/train/src_repo/ev_detection/train_det.py \
# --mode yolov5 --data /project/train/src_repo/ev_detection/data/EVDATA.yaml \
# --exist-ok --cfg /project/train/src_repo/ev_detection/modelYaml/yolov7.yaml \
# --weights /project/train/src_repo/ev_detection/yolov7.pt \
# --batch-size 16 --project /project/train/models --epochs 150 \
# --hyp /project/train/src_repo/ev_detection/data/hyps/hyp.scratch.yaml