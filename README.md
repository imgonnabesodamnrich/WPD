# 【论文笔记】基于小波包失真与卷积神经网络的机械系统极度失衡故障诊断

> **Mechanical Fault Diagnosis via WPD-CNN**
>
> **Paper Title:** *Highly imbalanced fault diagnosis of mechanical systems based on wavelet packet distortion and convolutional neural networks*
>
> **Journal:** *Advanced Engineering Informatics*, Vol. 51, 2022, 101535
>
> **DOI:** [10.1016/j.aei.2022.101535](https://doi.org/10.1016/j.aei.2022.101535)

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.aei.2022.101535-blue)](https://doi.org/10.1016/j.aei.2022.101535)
[![Status](https://img.shields.io/badge/Status-Published-brightgreen)](#)
[![Topic](https://img.shields.io/badge/Topic-Imbalanced%20Learning-orange)](#)

## 目录

* [1. 研究背景](#1-研究背景)
* [2. 技术原理：小波包失真（Wavelet Packet Distortion）](#2-技术原理小波包失真wavelet-packet-distortion)
    * [2.1 信号分解与变换](#21-信号分解与变换)
    * [2.2 非线性失真处理](#22-非线性失真处理)
    * [2.3 信号重构](#23-信号重构)
* [3. 分类模型架构：ConvNet](#3-分类模型架构convnet)
* [4. 实验验证](#4-实验验证)
    * [4.1 实验设置](#41-实验设置)
    * [4.2 性能对比](#42-性能对比)
* [5. 结论](#5-结论)
* [文献来源](#文献来源)

---


### 1. 研究背景

在机械系统监控领域，自动故障诊断技术对于保障工业安全具有重要意义。然而，实际工程场景中面临的核心挑战是**数据类别失衡（Class Imbalance）**：由于机械设备长时间处于正常运行状态，获取的健康样本充足，而真实的故障样本极度稀缺。

传统的深度学习模型在极度不平衡的数据集（如健康样本与故障样本比例达 80:1）上训练时，往往表现出明显的分类偏差，模型倾向于将所有样本预测为多数类（健康类），从而导致故障漏报。现有的数据增强方法（如随机过采样或合成少数类过采样）在处理极少量的故障样本时，容易造成过拟合或引入无关噪声。

### 2. 技术原理：小波包失真（Wavelet Packet Distortion）

论文提出了一种基于小波包失真（WPD）的数据增强策略，通过在时频域对原始信号进行物理意义明确的变换，生成具有多样性的新样本。

<div align="center">
  <img src="https://github.com/user-attachments/assets/9c1c7353-00f0-4f8d-8ea9-bdf849c986c1" width=70%/>
  <p><em>图1 小波包失真流程图</em></p>
</div>

#### 2.1 信号分解与变换
该方法首先利用小波包变换（Wavelet Packet Transform, WPT）将原始振动信号分解为不同频段的小波包系数。相比于标准小波变换，小波包变换能提供更精细的频率分辨率，从而更完整地捕捉故障特征。

#### 2.2 非线性失真处理
在分解后的系数空间中，随机选取特定的节点进行非线性处理。处理函数定义如下：
$$\tilde{w} = \text{sign}(w) \cdot (\text{abs}(w))^d$$

其中，w 为原始小波系数，d 为失真系数，其取值在预设范围内随机选取（如 [0.8, 1.2]）。

这种设计的物理逻辑在于：故障特征通常与特定的频率分量相关。通过调整 d 的数值，可以改变波形的具体形态（幅值细节），但能较好地保留信号原始的频率分布特性。这意味着生成的增强样本在保持故障类别属性的同时，增加了样本的空间分布多样性。

<div align="center">
  <img src="https://github.com/user-attachments/assets/a743b18c-7fcd-4569-aef5-a411939a02be" width=50%/>
  <p><em>图2 正弦信号及其失真版本</em></p>
</div>

#### 2.3 信号重构
通过小波包逆变换（Inverse WPT）将修改后的系数重构成时间序列信号。将生成的样本与原始样本混合，构建一个类别平衡的训练集。

### 3. 分类模型架构：ConvNet

论文采用了卷积神经网络（CNN）作为端到端的分类器。由于经过 WPD 增强后的数据集已经实现了类别平衡，CNN 可以有效地学习到区分不同健康状态的深层特征。模型包含卷积层、批量归一化层（BN）、ReLU 激活函数以及全局平均池化层，最终通过 Softmax 层输出分类结果。

<div align="center">
  <img src="https://github.com/user-attachments/assets/7edf8b67-6dac-4fc9-99d5-f0e02d324abb" width=50%/>
  <p><em>图3 整体诊断流程图</em></p>
</div>

### 4. 实验验证

#### 4.1 实验设置
实验数据来源于**民用航空液压泵**的地面仿真台架。数据集包含 1 类健康状态和 5 类故障状态。为了模拟极度失衡的情况，每类故障仅选取 5 个样本进行训练，而健康类包含 400 个训练样本。

| 标签 | 状态描述 | 训练样本数 | 测试样本数 |
| :--- | :--- | :--- | :--- |
| **H** | 正常运行状态 (No observable faults) | 400 | 80 |
| **F1** | 配油盘磨损故障 (Wear-out failure of oil distribution pan) | 5 | 475 |
| **F2** | 缸体磨损故障 (Wear-out failure of cylinder blocks) | 5 | 475 |
| **F3** | 低压入口冲蚀故障 (Eroding failure with 0.3 Mpa lower entry pressure) | 5 | 475 |
| **F4** | 出口压力异常导致的变形故障 (Distortion failure with 20 Mpa exit pressure) | 5 | 475 |
| **F5** | 0.8 mm 裂纹疲劳故障 (Fatigued failure with a chink of 0.8 mm) | 5 | 475 |

#### 4.2 性能对比
论文将该方法（Developed）与基准 CNN 模型、欠采样法（DS）以及简单过采样法（OS）进行了对比。

| 平均指标 (%) | ConvNet (基准) | DS + ConvNet (欠采样) | OS + ConvNet (过采样) | **Developed (论文提出的方法)** |
| :--- | :--- | :--- | :--- | :--- |
| **F1-score** | 89.40 ± 3.40 | 92.48 ± 2.95 | 96.03 ± 2.65 | **97.50 ± 1.74** |
| **Precision** | 91.10 ± 2.73 | 93.40 ± 2.63 | 96.66 ± 1.88 | **97.78 ± 1.27** |
| **Recall** | 89.52 ± 3.16 | 92.53 ± 2.97 | 96.10 ± 2.51 | **97.54 ± 1.65** |

*   **分类精度：** 在 10 次独立试验中，该方法的 F1 分数、精确率（Precision）和召回率（Recall）均表现出显著优势。平均 F1 分数达到 97.50%，相比于简单过采样方法提高了 1.47%。
*   **稳定性：** 箱线图分析显示，该方法的结果分布区间更窄，表明其对于不同批次的训练样本具有更强的鲁棒性。
*   **特征可视化：** 通过 t-SNE 降维分析发现，使用该方法训练的模型在特征空间中能够形成紧凑的类簇，且各类别之间的边界清晰，有效解决了特征重叠的问题。

### 5. 结论

小波包失真作为一种信号层面的数据增强手段，能够有效地缓解机械故障诊断中的小样本与不平衡问题。该方法计算效率高，不依赖复杂的生成网络，且生成的样本保留了原始物理特征，适用于民用航空、风力发电等关键工业领域的预测性维护任务。

---

### 文献来源

*   **标题:** Highly imbalanced fault diagnosis of mechanical systems based on wavelet packet distortion and convolutional neural networks
*   **期刊:** *Advanced Engineering Informatics*, 2022, 51: 101535.
*   **DOI:** [10.1016/j.aei.2022.101535](https://doi.org/10.1016/j.aei.2022.101535)
