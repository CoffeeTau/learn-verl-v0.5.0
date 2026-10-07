先建新索引：

```
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v3-retriever index
```

索引 `PASSED` 后，直接运行同集评测：

```
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v3-eval \
  --retriever-run main_20261007T042356Z_357492
```