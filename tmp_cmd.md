
**下一步：固定step125独立评测**

```
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v4-base-eval \
  --train-run main_20261009T100336Z_3e49ab \
  --expect-step 125
```

完成后再与旧V4的 **aligned自然80%、困难62%** 比较。目前一边是训练内验证、一边是独立评测，不能直接宣布“自然提高4个百分点、困难下降8个百分点”。