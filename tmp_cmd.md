**这是目标提醒＋GRPO 的可行性实验，尚不是带人工审核状态标签的 SFT。** 本地54项测试通过；服务器GPU运行尚待验证。实验日志和操作文档 (project/V4\_BASE\_RUNBOOK.md)已更新。

同步代码到服务器后，先跑8卡冒烟：

```
bash project/scripts/run.sh bash project/scripts/mainline.sh v4-base \
  --mode smoke --gpus 8
```

冒烟验收后，正式训练：

```
bash project/scripts/run.sh bash project/scripts/mainline.sh v4-base \
  --mode main --gpus 8
```