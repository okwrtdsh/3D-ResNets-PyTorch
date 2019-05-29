CUDA_VISIBLE_DEVICES=0,2 python main.py --root_path ../UCF101/ --video_path ../UCF101/jpg/ \
	--annotation_path ../UCF101/anotation/ucf101_01.json \
	--result_path results --dataset ucf101 --model resnet \
	--model_depth 34 --n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 1
