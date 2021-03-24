#python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ../data/kitti/training/velodyne/006837.bin
#python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth --data_path ../data/kitti/testing/velodyne/000008.bin
#python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 8 --ckpt pv_rcnn_8369.pth
#python train.py --cfg_file cfgs/kitti_models/second.yaml --epochs 1
#python train.py --cfg_file cfgs/kitti_models/HOS.yaml --epochs 1
python train.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --epochs 1
#python train.py --cfg_file cfgs/kitti_models/PartA2.yaml --epochs 1
