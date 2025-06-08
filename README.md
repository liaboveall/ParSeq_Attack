# 复现指南

---



## 环境配置

### IDE
推荐使用`PyChram`，用`vscode`可能遇到一些问题，包括但不限于可视化图表中的标签无法显示等等（但也有可能只是我的问题）

#

### 虚拟环境
推荐新建一个虚拟环境来进行复现，`venv`和`conda`都可以

#

### PyTorch 安装
推荐单独安装`Pytorch`，然后根据你是否有GPU来选择安装命令(`CUDA`和`cudnn`的安装自行解决吧)

#### venv 环境
如果你是venv：

**CPU 版本**
```bash
# CPU
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
```

**GPU 版本**
```bash
# CUDA
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

#

#### Conda 环境

**CPU 版本**
```bash
# CPU
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch
```

**GPU 版本**
```bash
# CUDA
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

#

### 其他依赖
缺省情况下这个项目的依赖需要查看`requirements`目录下的诸多txt文件，我稍微整合了一下，现在应该只需要`(在你安装完pytorch之后)`在本目录下执行:

```bash
pip install -r requirements.txt
```

---

#

## 复现 Parseq

### 用预训练模型
这个项目提有`torch hub`支持，可以先通过我的`torch_hub.py`脚本来拉取模型:

**列出所有可用模型**
```bash
# 列出所有模型
python torch_hub.py --list-models
```

**下载指定模型**
```bash
# 拉取
python torch_hub.py --model <model name>
```

#

### 图片识别
使用read.py

**示例命令**
```bash
# 例如：
python read.py pretrained=parseq --images ./CUTE80/image001.jpg --device cpu

# 具体参数自己看一下代码
```

---

#

# 攻击测试

## 📁 项目结构
所有攻击相关代码位于 `adversarial_attacks` 目录下

## 🎯 快速开始

### 📋 Demo 演示
```bash
# 运行 Jupyter Notebook 演示
jupyter notebook adversarial_attacks/demo.ipynb
```

### 🛠️ 攻击脚本使用

**🔥 推荐：AutoAttack（业界最强评估工具）**
- 集成多种先进攻击算法
- 专业的鲁棒性评估标准
- 详细的攻击效果分析

**📈 其他经典攻击算法**
- AdvGAN：基于生成对抗网络
- DeepFool：最小扰动攻击  
- 多攻击集成：FGSM、PGD等
- 增强版DeepFool：CUDA加速

#### 1️⃣ **AdvGAN 攻击**
```bash
# 位置：adversarial_attacks/advgan/
python adversarial_attacks/advgan/advgan.py
```
- **算法**：AdvGAN 生成对抗攻击
- **特点**：基于生成对抗网络的攻击方法

#### 2️⃣ **DeepFool 攻击**
```bash
# 位置：adversarial_attacks/deepfool/
python adversarial_attacks/deepfool/deepfool.py
```
- **算法**：DeepFool 最小扰动攻击
- **注意**：处理速度较慢（串行处理，未充分利用GPU）

#### 3️⃣ **多攻击选择**
```bash
# 位置：adversarial_attacks/multi/
python adversarial_attacks/multi/multi_attack.py
```
- **算法**：多种攻击算法可选择
- **包含**：FGSM、PGD、等多种经典攻击


#### 4️⃣ **advanced_deepfool 攻击**
```bash
# CUDA 加速版本（推荐）
python adversarial_attacks/advanced_deepfool/advanced_deepfool_cuda.py

# 普通版本（非常慢，不推荐）
python adversarial_attacks/advanced_deepfool/advanced_deepfool.py
```
- **算法**：advanced_deepfool 增强版攻击
- **特点**：CUDA 加速版本性能更优
- **注意**：普通版本未向量化，速度极慢

#### 5️⃣ **AutoAttack 攻击**
```bash
# 位置：adversarial_attacks/autoattack/
python adversarial_attacks/autoattack/autoattack_txt.py
```
- **算法**：AutoAttack 集成攻击框架
- **特点**：业界最强的对抗攻击评估工具
- **包含**：APGD-CE、APGD-DLR、FAB、Square等多种攻击
- **支持**：L∞ 和 L2 范数约束
- **优势**：专门适配文本识别模型，提供鲁棒性评估


**依赖安装：**
```bash
# 需要额外安装AutoAttack库
pip install git+https://github.com/fra31/auto-attack
```



