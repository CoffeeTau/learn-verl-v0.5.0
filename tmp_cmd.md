**① 先跑两步检查**

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v2 --mode smoke --gpus 8
```

截图最后的 `=== V2 | ... ===`，或：

```
runtime/runs/v2/latest_summary.txt
```

**② 如果 PASSED，可直接启动正式训练，无需等我再次确认**

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v2 --mode main --gpus 8
```

若检查显示 FAILED 或 NEEDS\_REVIEW，先回传摘要。正式训练重新从 V1 权重开始，不续接 smoke 权重