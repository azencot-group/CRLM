if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/CIFAR100" ]; then
    mkdir ./logs/CIFAR100
fi


for model_name in resNet18 resNet50 resNet101 VGG13 VGG16 VGG19
do
echo $model_name

    python -u train.py \
      --model $model_name \
      --dataset cifar100 > logs/CIFAR100/$model_name.log
done
