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
