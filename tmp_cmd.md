同步本次代码，**包括 `verl/` 目录的修改**，然后运行：

```
cd /home/h50061831/learn-verl-v0.5.0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash project/scripts/run.sh \
bash project/scripts/mainline.sh v4 --mode smoke --gpus 8
```