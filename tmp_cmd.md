现在运行：

```
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v4-eval \
  --train-run main_20261007T095646Z_3f4cdb
```

脚本会自动导出 **step 100**，评测自然200题及固定干预50题，不读取旧测试集。