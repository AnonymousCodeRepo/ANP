CUDA_VISIBLE_DEVICES="6" python main.py \
  --batchsize 128 \
  --epoch 120 \
  --lr 0.05 \
  --enable_lat 'True' \
  --test_flag 'True' \
  --test_data_path "/test/clean/test_data_cln.p" \
  --test_label_path "/test/clean/test_label.p" \
  --train_data_path "/cifar-10/" \
  --dataset "cifar10" \
  --model "vgg16" \
  --dropout 'True' \
  --layerlist "all" \
  --model_path "/vgg16_all/" \
  --logfile 'log/vgg16_all.txt' \
  --loss_acc_path 'loss_acc/vgg16_all/' \
  --enable_noise 'False' \
  --adv_flag 'False'

echo 'finish training...'
echo 'vgg-16 layer-all'
