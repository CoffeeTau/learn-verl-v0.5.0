```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash project/scripts/run.sh \
bash project/scripts/mainline.sh v4 --mode main --gpus 8
```

这会**从 V2 权重重新初始化**，不继承 smoke 的两步更新；训练125步，step 0及每25步评测自然200题和固定干预50题。