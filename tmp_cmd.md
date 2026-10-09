**固定 step100，只评测，不重新训练。**

```
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  python3 -m project.evaluation.v4 \
  --train-run main_20261007T095646Z_3f4cdb \
  --expect-step 100 --history-mode token-continuation
```
