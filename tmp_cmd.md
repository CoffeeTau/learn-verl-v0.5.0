**3\. 用四张空闲卡继续原批次**

```
bash project/scripts/run.sh \
  bash project/scripts/mainline.sh acceptance \
  --resume-latest --gpus 0,1,2,3
```

这里是**每张卡运行一个独立版本**，最多 V0–V3 同时运行。无需额外设置 `CUDA_VISIBLE_DEVICES`。