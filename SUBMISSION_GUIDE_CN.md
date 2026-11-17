# CMPE 401 Instructor-defined Project 2 — 中文提交说明

## 当前状态

- 实验已使用 `full` 模式完成；此文件夹整合了真实实验结果与模型源代码。
- Transformer / FordA：测试准确率 86.29%，不是 LSTM 的准确率。
- LSTM / Jena Climate：四组配置已完成，最佳方案为 learning rate 0.0005，验证 MSE 约 0.11136。
- 运行结果完整存放在 `outputs/` 目录，包括 2 张 PNG、1 份 CSV、1 份 JSON、1 份自动报告。

## GitHub 操作（不需要再训练）

1. 打开 https://github.com/ ，登录后创建一个新的公开或私人仓库（按老师要求决定可见性），例如 `CMPE401-Project2`。
2. 点击 **Add file → Upload files**，将本文件夹里面的文件及 `outputs/` 文件夹上传到仓库根目录，保留 `outputs/` 目录结构；不要只上传 ZIP 文件。
3. 在 GitHub 上打开 `README.md`，确认两张图片和实验表格正常显示。
4. 再检查 `CMPE401_Project2.ipynb`、`run_project.py`、`requirements.txt` 和实验结果文件是否完整。
5. 通读 README 中的两道 Reflection Questions，确保第一人称主观表述符合你自己的理解和真实经历；不符合就修改。
6. 最后按 Canvas 老师公布的提交方式上传 GitHub URL 或对应材料。原要求显示截止 12 月 13 日，最终以 Canvas 发布为准。

## 小心区分

- 本项目是 **Instructor-defined Project 2**，与占 40% 的 Self-defined Project 不同。
- 老师 PDF 对 Instructor-defined Project 2 要求的交付物是完整、说明清楚的 GitHub 代码与项目仓库。没有要求额外制作 Self-defined Project 的双栏报告或 3 分钟视频。
- README 明确写了 FordA 模型分类头相较于 Keras 官网示例做了 `Flatten()` 改动；不要称作逐行完全相同的官方代码。
- LSTM 的 MSE 和 MAE 属于 **验证集**，且目标标准化过；不能写成测试集结果或摄氏度误差。
- 源代码和复现说明可满足“训练模型文件或复现说明”的要求；当前 ZIP 不含训练权重。

## 最简提交检查表

- [ ] GitHub 有完整源代码（Notebook 与 Python script）
- [ ] GitHub 有 README 和运行说明
- [ ] GitHub 有 outputs 中 5 个真实结果文件
- [ ] Benchmark 表格显示 4 行（baseline + 三项改进）
- [ ] 两道问题已检查并按本人理解作答
- [ ] 打开 GitHub 链接，确认能看到项目内容
- [ ] Canvas 提交完成
