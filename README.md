# 复现指南

---



## 环境配置

### IDE
推荐使用`PyChram`，用`vscode`可能遇到一些问题，包括但不限于可视化图表中的标签无法显示等等（但也有可能只是我的问题）

#

### 虚拟环境
推荐新建一个虚拟环境来进行复现，`venv`和`conda`都可以

#

### 🔥 PyTorch 安装
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

## 攻击测试

- 查看`adversarial_attacks`目录

- 使用我的notebook demo

- 查看adversarial_attacks目录，里面每个目录下都有对应的攻击代码
  - advgan.py就是Adcgan的实现
  - deepfool.py就是DeepFool的实现，测试会有点慢，因为处理图像的方式还是串行的，其实和没用gpu是一样的
  - interactive_attack.py是他上课讲过的诸多算法的可选择的攻击实现（提示：不要选择C&W攻击）
  - superdeepfool_cuda.py是SuperDeepFool的实现,并进行了CUDA加速，superdeepfool.py没有向量化，会非常非常慢

