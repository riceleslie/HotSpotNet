#python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ../data/kitti/training/velodyne/006837.bin
#python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth --data_path ../data/kitti/testing/velodyne/000008.bin
#python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 8 --ckpt pv_rcnn_8369.pth
#python train.py --cfg_file cfgs/kitti_models/second.yaml --epochs 1
#python train.py --cfg_file cfgs/kitti_models/HOS.yaml --epochs 100
#python -m torch.distributed.launch --nproc_per_node=8 train.py --launcher pytorch --cfg_file cfgs/kitti_models/HOS.yaml
#python test.py --cfg_file cfgs/kitti_models/HOS.yaml --ckpt ../output/kitti_models/HOS/default/ckpt/checkpoint_epoch_100.pth 
#python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/HOS.yaml
#python train.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --epochs 1
#python train.py --cfg_file cfgs/kitti_models/PartA2.yaml --epochs 1
#python demo.py --cfg_file cfgs/kitti_models/HOS.yaml --ckpt ../output/kitti_models/HOS/default/ckpt/checkpoint_epoch_100.pth --data_path ../data/kitti/training/velodyne/006863.bin
#python demo.py --cfg_file cfgs/kitti_models/HOS.yaml --ckpt ./pointpillar_7728.pth --data_path ../data/kitti/training/velodyne/006863.bin
python demo.py --cfg_file cfgs/kitti_models/HOS.yaml --ckpt ../output/kitti_models/HOS/default/ckpt/checkpoint_epoch_100.pth --data_path 006863.bin
