# dase7506_mini_project

### 1. 训练模型
这里在 conda 环境下，cuda 版本为 2.7.1 进行训练。
在 `~/RIS/class_7506/MP1_student_starter/MP1_student_starter/code` 目录下运行：

```bash
python train.py --implementation student --device cuda --precision auto --seed 17 --eval-every 300 --run-dir runs/my-model-cuda-v11
```

### 2. 测试模型
运行测试命令，查看要求的 CPU 下的运行结果。
在 `~/RIS/class_7506/MP1_student_starter/MP1_student_starter/code` 目录下运行： 

```bash
/usr/bin/time -v -o runs/my-model-cuda-v11/test_cpu_fp32_resource.txt python evaluate.py --checkpoint runs/my-model-cuda-v11/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output runs/my-model-cuda-v11/test_cpu_fp32.json
```
> 注：这里省略了 vali 的评估，因为跟 test 没有关系。

### 3. 查看输出结果
得到评估结果、checkpoint、bpb 结果，以及 CPU 的利用报告。
在 `/home/coconut/RIS/class_7506/MP1_student_starter/MP1_student_starter/code/runs` 目录下，你将看到以下生成文件：

```text
checkpoint.pt
test_cpu_fp32.json
test_cpu_fp32_resource.txt
```
## AI Assistance Disclosure

本项目在开发和报告撰写过程中使用了生成式 AI 工具（ChatGPT）作为辅助。

AI assistance 主要用于以下方面：

- 帮助解释 Transformer、Attention、FFN、RoPE、QK-Norm、RMSNorm、SwiGLU、Dropout 等相关概念；
- 协助分析训练过程中记录的 gradient norm、residual update ratio、activation RMS 和 validation BPB 等诊断指标；
- 根据实验结果帮助提出和讨论可能的结构或训练策略改进方向；
- 协助检查部分 PyTorch 实现逻辑及代码修改，并协助实验结果进行整理和比较；
- 协助组织报告中的 Method、Ablation 和 Critical Analysis 的结构与语言表达。

对于 AI 提出的建议，我通过实际 ablation experiments 和 validation BPB 进行验证，而不是直接采用未经测试的结论。

我理解最终提交代码中所使用的主要模块及其作用，并能够解释模型结构、训练流程以及各项修改的实验依据。

除课程提供的 baseline code 和本 README 中明确说明的 AI assistance 外，本项目未直接复制未经注明的外部实现。
