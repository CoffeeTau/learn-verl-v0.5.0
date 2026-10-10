**下一步先做已保存结果的配对与案例检查，不需要重新训练或推理。**

```
bash project/scripts/run.sh python3 -m project.scripts.review_v4 \
  --base --eval-run dev_20261010T063219Z_9caa63
```

```
bash project/scripts/run.sh python3 -m project.scripts.cases_v4 \
  --stage base --eval-run dev_20261010T063219Z_9caa63
```