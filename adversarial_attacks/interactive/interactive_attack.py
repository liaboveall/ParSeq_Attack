#!/usr/bin/env python3
"""
CUTE80数据集对抗攻击可视化程序
支持交互式选择模型和攻击算法
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import warnings
import sys

# 添加父目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class AdversarialAttacker:
    """对抗攻击器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.eval()
        self.device = device
        
    def fgsm_attack(self, images, epsilon=0.1):
        """快速梯度符号方法 (FGSM) 攻击"""
        images = images.clone().detach().to(self.device)
        images.requires_grad = True
        
        # 前向传播获取logits
        logits = self.model(images)
        
        # 使用当前最可能的预测作为目标
        target_indices = logits.argmax(-1)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                              target_indices.view(-1))
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_images = images + epsilon * sign_data_grad
        
        # 限制像素值范围
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()
    
    def pgd_attack(self, images, epsilon=0.1, alpha=0.01, iters=10):
        """投影梯度下降 (PGD) 攻击"""
        images = images.clone().detach().to(self.device)
        ori_images = images.clone().detach()
        
        # 随机初始化扰动
        perturbed_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        for i in range(iters):
            perturbed_images.requires_grad = True
            
            logits = self.model(perturbed_images)
            target_indices = logits.argmax(-1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                  target_indices.view(-1))
            
            self.model.zero_grad()
            loss.backward()
            
            data_grad = perturbed_images.grad.data
            sign_data_grad = data_grad.sign()
            perturbed_images = perturbed_images.detach() + alpha * sign_data_grad
            
            # 投影到epsilon球内
            delta = torch.clamp(perturbed_images - ori_images, min=-epsilon, max=epsilon)
            perturbed_images = torch.clamp(ori_images + delta, 0, 1)
            
        return perturbed_images.detach()
    
    def c_w_attack(self, images, epsilon=0.1, c=1, kappa=0, max_iter=50, learning_rate=0.01):
        """Carlini & Wagner (C&W) 攻击 - 简化版本"""
        images = images.clone().detach().to(self.device)
        ori_images = images.clone().detach()
        
        # 初始化扰动
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=learning_rate)
        
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # 生成对抗样本
            perturbed_images = torch.clamp(ori_images + delta, 0, 1)
            
            # 前向传播
            logits = self.model(perturbed_images)
            
            # C&W损失函数
            real = logits.max(dim=-1)[0]  # 真实类别的logit
            other = logits.topk(2, dim=-1)[0][:, 1]  # 第二大的logit
            
            # f(x) = max(max_other - real + kappa, 0)
            f_loss = torch.clamp(other - real + kappa, min=0).mean()
            
            # L2距离损失
            l2_loss = torch.norm(delta, p=2, dim=[1, 2, 3]).mean()
            
            # 总损失
            total_loss = f_loss + c * l2_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 约束扰动大小
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        perturbed_images = torch.clamp(ori_images + delta, 0, 1)
        return perturbed_images.detach()


def predict_text(model, images):
    """使用模型的tokenizer正确解码文本"""
    with torch.no_grad():
        logits = model(images)
        probs = logits.softmax(-1)
        predictions, confidences = model.tokenizer.decode(probs)
    return predictions


def select_model():
    """交互式模型选择"""
    models = {
        1: {'name': 'parseq_tiny', 'desc': '微型模型 - 最快速度，适合快速测试'},
        2: {'name': 'parseq_patch16_224', 'desc': '标准模型 - 平衡性能和速度'},
        3: {'name': 'parseq', 'desc': '完整模型 - 最佳性能'},
        4: {'name': 'parseq_base', 'desc': '基础模型 - 标准配置'}
    }
    
    print("\n" + "="*60)
    print("可用的PARSeq模型:")
    print("="*60)
    for idx, model_info in models.items():
        print(f"{idx}. {model_info['name']}")
        print(f"   {model_info['desc']}")
        print("-" * 50)
    
    while True:
        try:
            choice = input("\n请选择模型 (输入数字 1-4): ").strip()
            choice = int(choice)
            
            if choice in models:
                selected = models[choice]
                print(f"\n✓ 已选择: {selected['name']}")
                print(f"  {selected['desc']}")
                return selected['name']
            else:
                print("❌ 无效选择，请输入 1-4 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认模型: parseq")
            return 'parseq'


def select_dataset_scope():
    """选择数据集范围"""
    scopes = {
        1: {'name': 'sample', 'desc': '小样本测试 - 前5张图像', 'use_all': False, 'num_images': 5},
        2: {'name': 'medium', 'desc': '中等规模测试 - 前20张图像', 'use_all': False, 'num_images': 20},
        3: {'name': 'large', 'desc': '大规模测试 - 前50张图像', 'use_all': False, 'num_images': 50},
        4: {'name': 'full', 'desc': '完整数据集 - 所有80张图像', 'use_all': True, 'num_images': 80}
    }
    
    print("\n" + "="*60)
    print("选择测试数据集范围:")
    print("="*60)
    for idx, scope_info in scopes.items():
        print(f"{idx}. {scope_info['name'].upper()}")
        print(f"   {scope_info['desc']}")
        print("-" * 50)
    
    while True:
        try:
            choice = input("\n请选择测试范围 (输入数字 1-4): ").strip()
            choice = int(choice)
            
            if choice in scopes:
                selected = scopes[choice]
                print(f"\n✓ 已选择: {selected['name'].upper()}")
                print(f"  {selected['desc']}")
                return selected['use_all'], selected['num_images']
            else:
                print("❌ 无效选择，请输入 1-4 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认设置: 小样本测试")
            return False, 5
def select_attack():
    """交互式攻击算法选择"""
    attacks = {
        1: {
            'name': 'FGSM',
            'desc': '快速梯度符号方法 - 单步攻击，速度最快',
            'params': {'epsilon': 0.1}
        },
        2: {
            'name': 'PGD',
            'desc': '投影梯度下降 - 多步迭代攻击，效果更好',
            'params': {'epsilon': 0.1, 'alpha': 0.02, 'iters': 10}
        },
        3: {
            'name': 'C&W',
            'desc': 'Carlini & Wagner攻击 - 优化基础攻击',
            'params': {'epsilon': 0.1, 'c': 1, 'max_iter': 30}
        },
        4: {
            'name': 'ALL',
            'desc': '执行所有攻击算法进行对比',
            'params': {}
        }
    }
    
    print("\n" + "="*60)
    print("可用的对抗攻击算法:")
    print("="*60)
    for idx, attack_info in attacks.items():
        print(f"{idx}. {attack_info['name']}")
        print(f"   {attack_info['desc']}")
        if attack_info['params']:
            param_str = ', '.join([f"{k}={v}" for k, v in attack_info['params'].items()])
            print(f"   参数: {param_str}")
        print("-" * 50)
    
    while True:
        try:
            choice = input("\n请选择攻击算法 (输入数字 1-4): ").strip()
            choice = int(choice)
            
            if choice in attacks:
                selected = attacks[choice]
                print(f"\n✓ 已选择: {selected['name']}")
                print(f"  {selected['desc']}")
                return choice, selected
            else:
                print("❌ 无效选择，请输入 1-4 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认攻击: FGSM")
            return 1, attacks[1]


def process_images_in_batches(image_files, model, batch_size=8):
    """分批处理图像以避免内存不足"""
    from strhub.data.module import SceneTextDataModule
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
    all_original_images = []
    all_image_tensors = []
    all_image_names = []
    
    print(f"\n正在分批预处理 {len(image_files)} 张图像 (批大小: {batch_size})...")
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        print(f"  处理批次 {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size} ({len(batch_files)} 张图像)")
        
        for img_path in batch_files:
            try:
                # 加载原始图像
                orig_img = Image.open(img_path).convert('RGB')
                all_original_images.append(orig_img)
                
                # 预处理
                img_tensor = transform(orig_img)
                all_image_tensors.append(img_tensor)
                
                # 获取图像名
                all_image_names.append(img_path.stem)
                
            except Exception as e:
                print(f"    警告: 加载图像 {img_path.name} 失败: {e}")
                continue
    
    if len(all_image_tensors) == 0:
        raise ValueError("没有成功加载任何图像！")
    
    print(f"成功预处理 {len(all_image_tensors)} 张图像")
    return all_original_images, all_image_tensors, all_image_names


def batch_attack_and_predict(attacker, model, image_tensors, attack_func, batch_size=8, **kwargs):
    """分批执行攻击和预测"""
    all_adv_images = []
    all_adv_texts = []
    
    num_batches = (len(image_tensors) + batch_size - 1) // batch_size
    print(f"  分 {num_batches} 个批次执行攻击...")
    
    for i in range(0, len(image_tensors), batch_size):
        batch_tensors = image_tensors[i:i+batch_size]
        batch = torch.stack(batch_tensors).to(device)
        
        # 执行攻击
        adv_batch = attack_func(batch, **kwargs)
        
        # 预测文本
        with torch.no_grad():
            batch_texts = predict_text(model, adv_batch)
        
        # 保存结果
        for j in range(adv_batch.size(0)):
            all_adv_images.append(adv_batch[j])
            all_adv_texts.append(batch_texts[j])
        
        print(f"    完成批次 {i//batch_size + 1}/{num_batches}")
    
    return torch.stack(all_adv_images), all_adv_texts


def load_cute80_images(use_all=False, num_images=5):
    """加载CUTE80数据集图像"""
    # 从当前脚本位置开始查找CUTE80目录
    current_script_dir = Path(__file__).parent
    
    # 尝试多个可能的路径
    possible_paths = [
        Path("CUTE80"),  # 当前工作目录下
        Path("../CUTE80"),  # 上一级目录
        Path("../../CUTE80"),  # 上两级目录（项目根目录）
        current_script_dir / "../../CUTE80",  # 相对于脚本位置的根目录
        Path(__file__).parent.parent.parent / "CUTE80"  # 绝对路径到根目录
    ]
    
    cute80_dir = None
    for path in possible_paths:
        if path.exists():
            cute80_dir = path
            break
    
    if cute80_dir is None:
        # 如果所有路径都不存在，显示详细的调试信息
        print("❌ 找不到CUTE80目录！")
        print(f"当前工作目录: {Path.cwd()}")
        print(f"脚本位置: {Path(__file__).parent}")
        print("尝试过的路径:")
        for i, path in enumerate(possible_paths, 1):
            print(f"  {i}. {path.absolute()} - {'存在' if path.exists() else '不存在'}")
        raise FileNotFoundError("未找到CUTE80数据集目录！请确保CUTE80文件夹在项目根目录下。")
        
    print(f"\n正在从 {cute80_dir.absolute()} 加载图像...")
    
    # 查找图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(list(cute80_dir.glob(ext)))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"在 {cute80_dir} 中未找到图像文件！")
    
    # 去重并排序
    unique_files = {}
    for file_path in image_files:
        key = str(file_path.resolve()).lower()
        if key not in unique_files:
            unique_files[key] = file_path
    
    image_files = sorted(unique_files.values())
    
    if use_all:
        print(f"使用整个数据集: {len(image_files)} 张图像")
    else:
        image_files = image_files[:num_images]
        print(f"使用前 {len(image_files)} 张图像")
    
    return image_files



def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化tensor用于显示"""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


def tensor_to_image(tensor):
    """将tensor转换为可显示的numpy数组"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)


def visualize_results(original_images, attack_results, original_texts, image_names, max_display=10):
    """可视化攻击结果 - 限制显示数量以避免图表过大"""
    num_images = min(len(original_images), max_display)
    num_attacks = len(attack_results)
    
    if num_images == 0:
        print("没有图像可以显示")
        return
    
    # 创建子图
    fig, axes = plt.subplots(num_images, num_attacks + 2, figsize=(4 * (num_attacks + 2), 3 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # 设置标题
    display_info = f"(显示前{num_images}张)" if len(original_images) > max_display else ""
    fig.suptitle(f'PARSeq文本识别模型对抗攻击结果 {display_info}', fontsize=16, fontweight='bold')
    
    # 列标题
    col_titles = ['原始图像'] + [f'{name}攻击' for name in attack_results.keys()] + ['扰动对比']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight='bold')
    
    for i in range(num_images):
        # 显示原始图像
        axes[i, 0].imshow(original_images[i])
        axes[i, 0].set_xlabel(f'识别: "{original_texts[i]}"', fontsize=10)
        axes[i, 0].set_ylabel(f'图像 {i+1}\n({image_names[i]})', fontsize=10)
        axes[i, 0].axis('off')
        
        # 显示攻击结果
        for j, (attack_name, attack_data) in enumerate(attack_results.items()):
            adv_images, adv_texts = attack_data
            
            # 反归一化并显示对抗样本
            adv_denorm = denormalize_tensor(adv_images[i:i+1])
            adv_img = tensor_to_image(adv_denorm[0])
            axes[i, j + 1].imshow(adv_img)
            
            # 显示识别结果和攻击状态
            success = original_texts[i] != adv_texts[i]
            status = "✓成功" if success else "✗失败"
            color = 'red' if success else 'green'
            
            axes[i, j + 1].set_xlabel(f'识别: "{adv_texts[i]}"\n状态: {status}', 
                                    fontsize=10, color=color)
            axes[i, j + 1].axis('off')
        
        # 显示扰动
        if len(attack_results) > 0:
            # 使用第一个攻击的扰动作为示例
            first_attack_name = list(attack_results.keys())[0]
            first_adv_images = attack_results[first_attack_name][0]
            
            # 计算扰动
            original_tensor = torch.stack([tensor_to_tensor(original_images[i])])
            perturbation = first_adv_images[i:i+1] - original_tensor.to(device)
            
            # 放大扰动以便观察
            perturbation_vis = torch.clamp((perturbation * 10 + 0.5), 0, 1)
            perturbation_img = tensor_to_image(perturbation_vis[0])
            
            axes[i, -1].imshow(perturbation_img)
            axes[i, -1].set_xlabel(f'{first_attack_name}扰动\n(已放大显示)', fontsize=10)
            axes[i, -1].axis('off')
    
    plt.tight_layout()
    plt.show()


def tensor_to_tensor(pil_image):
    """将PIL图像转换为tensor"""
    from strhub.data.module import SceneTextDataModule
    transform = SceneTextDataModule.get_transform((32, 128))  # 使用标准尺寸
    return transform(pil_image)


def print_attack_summary(attack_results, original_texts, image_names):
    """打印攻击总结"""
    print("\n" + "="*80)
    print("攻击结果总结")
    print("="*80)
    
    total_images = len(original_texts)
    print(f"总图像数量: {total_images}")
    
    for attack_name, (adv_images, adv_texts) in attack_results.items():
        success_count = sum(1 for orig, adv in zip(original_texts, adv_texts) if orig != adv)
        success_rate = success_count / len(original_texts)
        
        print(f"\n{attack_name} 攻击结果:")
        print(f"  成功率: {success_count}/{len(original_texts)} = {success_rate:.1%}")
        print(f"  失败率: {len(original_texts) - success_count}/{len(original_texts)} = {1-success_rate:.1%}")
        
        # 显示部分详细结果
        print(f"  详细结果 (显示前10个):")
        for i, (orig, adv, name) in enumerate(zip(original_texts[:10], adv_texts[:10], image_names[:10])):
            status = "✓" if orig != adv else "✗"
            print(f"    {i+1:2d}. {name}: \"{orig}\" -> \"{adv}\" {status}")
        
        if total_images > 10:
            print(f"    ... (还有 {total_images - 10} 个结果)")
    
    print("="*80)


def main():
    """主函数"""
    print("="*60)
    print("PARSeq 文本识别模型对抗攻击演示程序")
    print("="*60)
    
    try:
        # 1. 选择模型
        model_name = select_model()
        
        # 2. 加载模型
        print(f"\n正在加载模型: {model_name}...")
        model = torch.hub.load('baudm/parseq', model_name, pretrained=True, trust_repo=True)
        model.eval()
        model.to(device)
        print(f"✓ 模型 {model_name} 加载成功!")
        
        # 3. 选择数据集范围
        use_all, num_images = select_dataset_scope()
        
        # 4. 加载图像
        image_files = load_cute80_images(use_all=use_all, num_images=num_images)
        
        # 分批处理大数据集
        batch_size = 8 if len(image_files) > 20 else len(image_files)
        original_images, image_tensors, image_names = process_images_in_batches(
            image_files, model, batch_size=batch_size)
        
        # 5. 原始识别 - 分批进行
        print(f"\n正在进行原始文本识别 ({len(image_tensors)} 张图像)...")
        original_texts = []
        
        for i in range(0, len(image_tensors), batch_size):
            batch_tensors = image_tensors[i:i+batch_size]
            batch = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                batch_texts = predict_text(model, batch)
                original_texts.extend(batch_texts)
            
            print(f"  完成 {min(i+batch_size, len(image_tensors))}/{len(image_tensors)} 张图像的识别")
        
        print("\n原始识别结果 (显示前10个):")
        for i, (name, text) in enumerate(zip(image_names[:10], original_texts[:10])):
            print(f"  图像 {i+1} ({name}): \"{text}\"")
        if len(original_texts) > 10:
            print(f"  ... (还有 {len(original_texts) - 10} 个结果)")
        
        # 6. 选择攻击算法
        attack_choice, attack_info = select_attack()
        
        # 7. 执行攻击
        print(f"\n正在执行 {attack_info['name']} 攻击...")
        attacker = AdversarialAttacker(model, device)
        attack_results = {}
        
        if attack_choice == 1:  # FGSM
            print("执行 FGSM 攻击...")
            adv_images, adv_texts = batch_attack_and_predict(
                attacker, model, image_tensors, attacker.fgsm_attack, 
                batch_size=batch_size, epsilon=0.1)
            attack_results['FGSM'] = (adv_images, adv_texts)
            
        elif attack_choice == 2:  # PGD
            print("执行 PGD 攻击...")
            adv_images, adv_texts = batch_attack_and_predict(
                attacker, model, image_tensors, attacker.pgd_attack, 
                batch_size=batch_size, epsilon=0.1, alpha=0.02, iters=10)
            attack_results['PGD'] = (adv_images, adv_texts)
            
        elif attack_choice == 3:  # C&W
            print("执行 C&W 攻击...")
            adv_images, adv_texts = batch_attack_and_predict(
                attacker, model, image_tensors, attacker.c_w_attack, 
                batch_size=batch_size, epsilon=0.1, c=1, max_iter=30)
            attack_results['C&W'] = (adv_images, adv_texts)
            
        elif attack_choice == 4:  # 所有攻击
            print("执行 FGSM 攻击...")
            fgsm_adv, fgsm_texts = batch_attack_and_predict(
                attacker, model, image_tensors, attacker.fgsm_attack, 
                batch_size=batch_size, epsilon=0.1)
            attack_results['FGSM'] = (fgsm_adv, fgsm_texts)
            
            print("执行 PGD 攻击...")
            pgd_adv, pgd_texts = batch_attack_and_predict(
                attacker, model, image_tensors, attacker.pgd_attack, 
                batch_size=batch_size, epsilon=0.1, alpha=0.02, iters=10)
            attack_results['PGD'] = (pgd_adv, pgd_texts)
            
            print("执行 C&W 攻击...")
            cw_adv, cw_texts = batch_attack_and_predict(
                attacker, model, image_tensors, attacker.c_w_attack, 
                batch_size=batch_size, epsilon=0.1, c=1, max_iter=30)
            attack_results['C&W'] = (cw_adv, cw_texts)
        
        # 8. 显示结果
        print_attack_summary(attack_results, original_texts, image_names)
        
        # 9. 可视化 (仅显示前几张图像)
        if len(original_images) > 0:
            print("\n正在生成可视化结果...")
            max_display = min(10, len(original_images))
            print(f"(将显示前 {max_display} 张图像的可视化结果)")
            visualize_results(original_images, attack_results, original_texts, image_names, max_display=max_display)
        
        # 10. 保存结果统计
        save_results_summary(model_name, attack_results, original_texts, image_names, len(image_files))
        
        print(f"\n✓ 在 {len(image_files)} 张图像上的攻击演示完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


def save_results_summary(model_name, attack_results, original_texts, image_names, total_images):
    """保存结果统计到文件"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attack_results_{model_name}_{total_images}images_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"PARSeq对抗攻击结果报告\n")
            f.write(f"="*50 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"总图像数: {total_images}\n")
            f.write(f"设备: {device}\n\n")
            
            for attack_name, (adv_images, adv_texts) in attack_results.items():
                success_count = sum(1 for orig, adv in zip(original_texts, adv_texts) if orig != adv)
                success_rate = success_count / len(original_texts)
                
                f.write(f"{attack_name} 攻击结果:\n")
                f.write(f"  成功率: {success_count}/{len(original_texts)} = {success_rate:.2%}\n")
                f.write(f"  失败率: {len(original_texts) - success_count}/{len(original_texts)} = {1-success_rate:.2%}\n\n")
                
                f.write(f"  详细结果:\n")
                for i, (orig, adv, name) in enumerate(zip(original_texts, adv_texts, image_names)):
                    status = "SUCCESS" if orig != adv else "FAILED"
                    f.write(f"    {i+1:3d}. {name}: \"{orig}\" -> \"{adv}\" [{status}]\n")
                f.write("\n")
        
        print(f"✓ 结果已保存到: {filename}")
        
    except Exception as e:
        print(f"❌ 保存结果文件失败: {e}")


if __name__ == "__main__":
    main()
