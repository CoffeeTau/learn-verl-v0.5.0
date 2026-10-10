**下一步先看生日题的完整自然／困难轨迹：**

```
bash project/scripts/run.sh python3 -m project.scripts.cases_v4 \
  --stage base \
  --eval-run dev_20261010T063219Z_9caa63 \
  --case-id 199fbba20bdc11eba7f7acde48001122
```

再从本次 `badcase_evidence.md` 搜索 **`Some Came Running`**&#65292;查看 `V4-base-hard` 的逐轮输出和证据。优先发这两个案例的轨迹，能帮助我们决定下一轮应监督“人物身份核对”还是“缺失属性补查”。