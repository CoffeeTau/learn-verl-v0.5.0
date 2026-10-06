**1\. 在 VS Code 的远端终端检查依赖**

进入项目根目录，使用原训练环境：

```
bash project/scripts/run.sh python3 -c \
  "from torch.utils.tensorboard import SummaryWriter; import tensorboard; print('TensorBoard', tensorboard.__version__)"
```

如果提示缺少 `tensorboard`，再执行：

```
bash project/scripts/run.sh python3 -m pip install tensorboard
```

**2\. 导入这次训练与评测结果**

```
bash project/scripts/run.sh python3 -m project.scripts.tensorboard_export \
  --train-run main_20261006T090840Z_c41e35 \
  --eval-run dev_20261006T151952Z_111fcb
```

脚本会把 JSONL 转成 TensorBoard 事件文件，保存到：

```
runtime/tensorboard/v1_main_20261006T090840Z_c41e35/
```

这是一次历史导入，**目前不会持续监听 JSONL**。后续训练的实时记录，我们再接入。

**3\. 启动 TensorBoard**

```
bash project/scripts/run.sh bash project/scripts/tensorboard.sh
```

保持这个终端运行。TensorBoard 会读取事件文件并提供浏览器页面。[官方说明 (https://github.com/tensorflow/tensorboard)](<https://github.com/tensorflow/tensorboard>)

**4\. 在 VS Code 转发端口**

在已连接服务器的 VS Code 窗口中：

1. 打开底部的 **“端口 / PORTS”** 面板。
2. 点击 **“转发端口 / Forward a Port”**&#65292;输入 `6006`。
3. 点击这一行的 **“在浏览器中打开”** 图标。

找不到面板时，在命令面板搜索 `Ports: Focus on Ports View`。通常浏览器地址是 `http://localhost:6006`，以面板显示的本地地址为准。[VS Code 官方说明 (https://code.visualstudio.com/docs/remote/ssh\#\_forwarding-a-port-creating-ssh-tunnel)](<https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnel>)