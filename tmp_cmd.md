不必为了与训练内数字完全相同继续反复调参。下一步用已有脚本做一次**只读逐题配对检查**：

```
bash project/scripts/run.sh python3 -m project.scripts.review_v4 \
  --aligned --eval-run dev_20261009T064354Z_7f3ed6
```

