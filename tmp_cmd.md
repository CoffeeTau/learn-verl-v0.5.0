同步代码后，**不用重训 E5、不用重建索引**，先运行：

```
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v3-eval \
  --retriever-run main_20261007T042356Z_357492 --smoke
```