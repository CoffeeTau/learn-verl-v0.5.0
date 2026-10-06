# TensorBoard：VS Code Remote SSH

本轮先展示已完成 V1 的历史曲线，不重训。路径：`metrics.jsonl → 导入脚本 → events.out.tfevents.* → TensorBoard → SSH 端口转发 → 本地浏览器`。历史导入不是持续监听；后续训练实时写入及定期 dev 评测尚未在本轮配置。

在远端仓库根目录、原训练 Python 环境中执行。先同步新增脚本。

```bash
bash project/scripts/run.sh python3 -c "from torch.utils.tensorboard import SummaryWriter; import tensorboard; print('TensorBoard', tensorboard.__version__)"
```

只有提示缺少 tensorboard 时才安装（不升级 torch/transformers/vLLM）：

```bash
bash project/scripts/run.sh python3 -m pip install tensorboard
```

导入完成的训练与对应评测：

```bash
bash project/scripts/run.sh python3 -m project.scripts.tensorboard_export \
  --train-run main_20261006T090840Z_c41e35 \
  --eval-run dev_20261006T151952Z_111fcb
```

生成 `runtime/tensorboard/v1_main_20261006T090840Z_c41e35/`，包含事件文件和 import_manifest.json；重复执行校验来源后复用。当前 wall time 是导入时间，查看曲线请选择 Step 横轴。

启动服务并保持终端打开：

```bash
bash project/scripts/run.sh bash project/scripts/tensorboard.sh
```

在 VS Code 已连接远端的窗口中：底部“端口 / PORTS”→“转发端口 / Forward a Port”→输入 `6006`→点击“在浏览器中打开”。找不到面板时，在命令面板搜索 `Ports: Focus on Ports View`。使用面板给出的本地地址；6006 已被占用时本地端口可能不同。服务只监听远端 127.0.0.1，通过现有 SSH 连接访问，不需要公开端口。

在 Scalars 页面选择该 run，先将 Smoothing 设为 0。优先看：

- `train/answer_f1`：每步训练批次平均 F1，范围 0–1；这是 `critic/score/mean` 的易读别名。
- `actor/grad_norm`、`actor/kl_loss`（若日志中存在）：更新稳定性与参考策略偏离。
- `timing_s/step`：每步耗时。
- `eval/em_percent`、`eval/f1_percent`：百分制，仅 step 0 与 125 两个真实测量点。连接线不代表中间评测结果，且与训练 F1 的量纲不同。

回传：导入脚本的短摘要、浏览器中的 `train/answer_f1` 和 `eval/em_percent` 曲线截图。若打不开页面，截图服务终端最后几行与 VS Code 的 PORTS 面板。无需回传完整事件文件。

本地仅检查了标量转换（无虚构中间 dev 点、过滤非有限指标）和 shell 语法；实际事件文件生成及页面显示需要远端验证。本轮不写入任何虚构训练指标。
