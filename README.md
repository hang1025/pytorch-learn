# 📘 pytorch-learn · 动手学深度学习（PyTorch 版）学习笔记

> 跟着李沐老师《动手学深度学习》(Dive into Deep Learning, d2l) **PyTorch 版**，从零开始系统学习深度学习。
> 本仓库为全书代码 + 个人学习环境记录，边学边更新。

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0%2Bcu113-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9-3776ab?logo=python)](https://www.python.org/)
[![d2l](https://img.shields.io/badge/d2l-0.17.6-orange)](https://zh.d2l.ai/)
[![GPU](https://img.shields.io/badge/GPU-RTX%203050-76b900?logo=nvidia)](#)

---

## 📖 关于本项目

本书配套代码来自 [zh.d2l.ai](https://zh.d2l.ai/chapter_installation/index.html)，提供 **MXNet / PyTorch / TensorFlow / Paddle** 四个框架版本，本仓库使用的是 **PyTorch 版**。

学习目标：掌握深度学习的核心概念（张量、自动微分、神经网络、CNN、RNN、Transformer…），并能用 PyTorch 亲手实现。

---

## 🗂️ 章节地图

每个文件夹对应书的一章，按学习顺序排列：

| 文件夹 | 章节 | 主要内容 |
|---|---|---|
| `chapter_introduction` | 第 1 章 · 引言 | 深度学习是什么 |
| `chapter_preliminaries` | 第 2 章 · 预备知识 ⭐ | 张量、自动微分、数学基础 |
| `chapter_linear-networks` | 第 3 章 · 线性神经网络 | 线性回归、softmax |
| `chapter_multilayer-perceptrons` | 第 4 章 · 多层感知机 | MLP、过拟合、Dropout |
| `chapter_deep-learning-computation` | 第 5 章 · 深度学习计算 | 模型构造、参数管理、GPU |
| `chapter_convolutional-neural-networks` | 第 6 章 · 卷积神经网络 | CNN、LeNet |
| `chapter_convolutional-modern` | 第 7 章 · 现代 CNN | AlexNet / VGG / ResNet / DenseNet |
| `chapter_recurrent-neural-networks` | 第 8 章 · 循环神经网络 | RNN、语言模型 |
| `chapter_recurrent-modern` | 第 9 章 · 现代 RNN | GRU、LSTM、seq2seq |
| `chapter_attention-mechanisms` | 第 10 章 · 注意力机制 | Attention、Transformer |
| `chapter_optimization` | 第 11 章 · 优化算法 | SGD、Adam、学习率调度 |
| `chapter_computational-performance` | 第 12 章 · 计算性能 | 异步计算、多 GPU |
| `chapter_computer-vision` | 第 13 章 · 计算机视觉 | 目标检测、语义分割、风格迁移 |
| `chapter_natural-language-processing-pretraining` | 第 14 章 · NLP 预训练 | word2vec、BERT |
| `chapter_natural-language-processing-applications` | 第 15 章 · NLP 应用 | 情感分析、自然语言推断 |
| `chapter_appendix-tools-for-deep-learning` | 第 16 章 · 附录 | Jupyter、云服务器等工具 |

其他文件：
- `img/` —— 书中插图与数据集样例
- `TERMINOLOGY.ipynb` —— 中英术语对照表
- `setup.py` —— d2l 包安装脚本
- `d2l.bib` —— 参考文献

---

## 🛠️ 环境配置（Windows + NVIDIA GPU）

> 我的硬件：**Windows 11 + NVIDIA GeForce RTX 3050 Laptop GPU (4GB)**

### 1. 安装 Anaconda / Miniconda
已有 conda 环境管理工具即可（本项目使用 Anaconda）。

### 2. 创建独立环境（Python 3.9）
```bash
conda create --name d2l python=3.9 -y
conda activate d2l
```
> ⚠️ 本书代码基于 **Python 3.9 + PyTorch 1.12** 编写，请勿使用过高版本的 Python，否则旧版 torch 无法安装。

### 3. 安装 PyTorch（GPU 版，CUDA 11.3）
```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
> ⚠️ `+cu113` 和 `--extra-index-url` **不能省略**，否则会装成 CPU 版，无法使用显卡加速。
> CUDA 版本需与显卡驱动匹配（RTX 3050 驱动 ≥ 465 即可支持 cu113）。

### 4. 安装 d2l 与 Jupyter
```bash
pip install d2l==0.17.6
pip install jupyter
```

### 5. 验证环境
```python
import torch
print(torch.__version__)          # 1.12.0+cu113
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 3050 Laptop GPU

import d2l
print(d2l.__version__)            # 0.17.6
```

---

## ▶️ 如何运行

本书代码均为 Jupyter Notebook（`.ipynb`），有两种运行方式：

### 方式一：PyCharm Professional（本项目使用）
1. 用 PyCharm 打开本仓库根目录
2. 将解释器设为 d2l 环境：`Settings → Python Interpreter → Conda → Existing → d2l`
   （或 System Interpreter 指向 `D:\Anaconda\envs\d2l\python.exe`）
3. 双击任意 `.ipynb` 文件，PyCharm 内置 Jupyter 即可逐格运行（`Shift + Enter`）

### 方式二：浏览器 Jupyter
```bash
conda activate d2l
cd pytorch            # 进入对应框架目录
jupyter notebook      # 或 jupyter lab
```
浏览器访问 `http://localhost:8888` 即可。

---

## 🗺️ 学习路线建议

```
第2章 预备知识（张量 + 自动微分是地基）
   ↓
第3~4章 线性网络 / MLP（第一个真正的模型）
   ↓
第5章 深度学习计算（学会搭模型、用 GPU）
   ↓
第6~7章 CNN（图像方向核心）
   ↓
第8~9章 RNN（序列 / 文本方向）
   ↓
第10章 注意力机制 / Transformer（现代大模型基础）
   ↓
第11~15章 优化 / 性能 / CV / NLP 实战
```

> 💡 建议从 `chapter_preliminaries/ndarray.ipynb`（张量）和 `autograd.ipynb`（自动微分）开始，这两节是后续所有章节的基础。

---

## 📝 学习进度

- [x] 环境搭建（Python 3.9 + PyTorch 1.12+cu113 + GPU 验证通过）
- [ ] 第 2 章 预备知识
- [ ] 第 3 章 线性神经网络
- [ ] 第 4 章 多层感知机
- [ ] ……（持续更新）

---

## 🙏 致谢

- 教材：《动手学深度学习》—— [zh.d2l.ai](https://zh.d2l.ai/) ｜ [GitHub](https://github.com/d2l-ai/d2l-zh)
- 框架：[PyTorch](https://pytorch.org/)

---

## 📌 小贴士

- 每次在命令行运行前记得先 `conda activate d2l`；在 PyCharm 中设好解释器后则无需手动激活。
- 国内下载 PyTorch 较慢时，可考虑使用镜像源。
- `.idea/` 为 PyCharm 配置目录，可按需加入 `.gitignore`。
