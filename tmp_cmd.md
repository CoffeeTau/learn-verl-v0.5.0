**当前最值得注意：V4 的困难题退化不能主要用格式错误解释。** 同时，`answered` 也不等于“自信答错”，里面可能包含拒答，需要逐题看。

我已补好只读提取脚本，并更新实验日志及案例文档。代码同步到服务器后运行：

```
bash project/scripts/run.sh python3 -m project.scripts.cases_v4 \
  --eval-run dev_20261007T160345Z_37aa80
```