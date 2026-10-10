```
bash project/scripts/run.sh python3 -m project.scripts.cases_v4 \
  --stage base \
  --eval-run dev_20261010T063219Z_9caa63 \
  --case-id 199fbba20bdc11eba7f7acde48001122 \
  > runtime/birthday_v4_base.txt
```

然后只打印V4-base的结果摘要：

```
rg '^V4-base' runtime/birthday_v4_base.txt
```

会显示自然／困难两行的 **EM、搜索次数、状态和预测答案**。

完整文件在：

```
runtime/birthday_v4_base.txt
```