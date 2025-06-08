#!/usr/bin/env python3
"""
SuperDeepFool的主要改进：
1. 自适应步长控制
2. 多类别决策边界优化
3. 梯度动量加速
4. 早停策略
5. 动态超调参数调整
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import warnings
import sys
import time
from collections import deque

# 添加父目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def predict_text(model, images):
    """使用模型的tokenizer正确解码文本"""
    with torch.no_grad():
        logits = model(images)
        probs = logits.softmax(-1)
        predictions, confidences = model.tokenizer.decode(probs)
    return predictions


class SuperDeepFoolAttacker:
    """SuperDeepFool对抗攻击器 - 改进版DeepFool算法"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.eval()
        self.device = device
        
        # SuperDeepFool特有参数
        self.momentum = 0.9
        self.adaptive_overshoot = True
        self.top_k_classes = 5
        self.convergence_threshold = 1e-6
        self.min_overshoot = 0.01
        self.max_overshoot = 0.1
        
    def superdeepfool_attack(self, images, max_iter=100, overshoot=0.02):
        """
        SuperDeepFool攻击实现
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            max_iter: 最大迭代次数
            overshoot: 初始超调参数
            
        Returns:
            adversarial_images: 对抗样本
            perturbations: 扰动
            iterations: 实际迭代次数
            convergence_history: 收敛历史
        """
        images = images.clone().detach().to(self.device)
        batch_size = images.shape[0]
        
        adversarial_images = []
        perturbations = []
        iterations = []
        convergence_histories = []
        
        print(f"开始SuperDeepFool攻击 (批大小: {batch_size}, 最大迭代: {max_iter})...")
        
        for batch_idx in range(batch_size):
            print(f"  处理图像 {batch_idx + 1}/{batch_size}...")
            
            x = images[batch_idx:batch_idx+1].clone()
            r_total = torch.zeros_like(x)
            
            # 获取原始预测
            with torch.no_grad():
                orig_logits = self.model(x)
                orig_pred_text = predict_text(self.model, x)[0]
            
            # SuperDeepFool特有变量
            momentum_buffer = torch.zeros_like(x)
            current_overshoot = overshoot
            convergence_history = []
            patience_counter = 0
            best_perturbation = float('inf')
            no_improvement_count = 0
            
            start_time = time.time()
            
            for i in range(max_iter):
                x_adv = x + r_total
                x_adv.requires_grad_(True)
                
                # 前向传播
                logits = self.model(x_adv)
                
                # 每5轮检查一次预测变化（减少频繁检查）
                if i % 5 == 0:
                    with torch.no_grad():
                        current_pred_text = predict_text(self.model, x_adv)[0]
                        if current_pred_text != orig_pred_text:
                            iterations.append(i + 1)
                            break
                
                # 计算多类别梯度
                grad_dict = self._compute_multiclass_gradients(x_adv, logits)
                
                # 选择最优扰动方向
                optimal_perturbation = self._select_optimal_perturbation(grad_dict, current_overshoot)
                
                # 应用动量
                momentum_buffer = self.momentum * momentum_buffer + (1 - self.momentum) * optimal_perturbation
                
                # 更新扰动
                r_total = r_total + momentum_buffer
                
                # 计算当前扰动大小
                current_norm = torch.norm(r_total).item()
                convergence_history.append(current_norm)
                
                # 自适应调整超调参数
                if self.adaptive_overshoot:
                    current_overshoot = self._adaptive_overshoot_update(
                        current_norm, best_perturbation, current_overshoot, i
                    )
                
                # 早停检查（减少检查频率）
                if i > 10 and i % 5 == 0 and self._check_convergence(convergence_history):
                    print(f"    收敛检测: 第 {i+1} 轮提前停止")
                    iterations.append(i + 1)
                    break
                
                # 更新最佳扰动
                if current_norm < best_perturbation:
                    best_perturbation = current_norm
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # 耐心机制
                if no_improvement_count > 10:
                    current_overshoot = min(current_overshoot * 1.1, self.max_overshoot)
                    no_improvement_count = 0
                
                # 进度显示
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"    轮次 {i+1:3d}: 扰动L2={current_norm:.6f}, "
                          f"超调={current_overshoot:.4f}, 耗时={elapsed:.1f}s")
            else:
                # 达到最大迭代次数
                iterations.append(max_iter)
            
            # 生成最终对抗样本
            final_adv = torch.clamp(x + r_total, 0, 1)
            adversarial_images.append(final_adv.squeeze(0))
            perturbations.append(r_total.squeeze(0))
            convergence_histories.append(convergence_history)
            
            elapsed = time.time() - start_time
            print(f"    完成 (迭代: {iterations[-1]}, 最终L2范数: {torch.norm(r_total).item():.6f}, "
                  f"总耗时: {elapsed:.1f}s)")
        
        adversarial_images = torch.stack(adversarial_images)
        perturbations = torch.stack(perturbations)
        
        print("SuperDeepFool攻击完成!")
        return adversarial_images.detach(), perturbations.detach(), iterations, convergence_histories
    
    def _compute_multiclass_gradients(self, x_adv, logits):
        """计算多类别梯度 - 优化版本"""
        seq_len, vocab_size = logits.shape[1], logits.shape[2]
        grad_dict = {}
        
        # 获取当前预测的top-k类别（减少k值以提高速度）
        current_logits = logits.view(-1, vocab_size)
        topk_k = min(3, self.top_k_classes)  # 限制为最多3个类别
        topk_values, topk_indices = torch.topk(current_logits, topk_k, dim=-1)
        
        # 只处理前几个时间步以提高速度
        max_time_steps = min(seq_len, 8)  # 限制时间步数
        
        for t in range(max_time_steps):
            current_class = topk_indices[t, 0]
            
            # 一次性计算所有需要的梯度
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
                
            # 计算当前类别的损失
            loss_current = current_logits[t, current_class]
            loss_current.backward(retain_graph=True)
            grad_current = x_adv.grad.clone() if x_adv.grad is not None else torch.zeros_like(x_adv)
            
            # 只处理最重要的1-2个其他类别
            for k in range(1, min(2, topk_k)):  # 减少处理的类别数
                if k < topk_indices.shape[1]:
                    other_class = topk_indices[t, k]
                    
                    self.model.zero_grad()
                    if x_adv.grad is not None:
                        x_adv.grad.zero_()
                    
                    loss_other = current_logits[t, other_class]
                    loss_other.backward(retain_graph=True)
                    grad_other = x_adv.grad.clone() if x_adv.grad is not None else torch.zeros_like(x_adv)
                    
                    # 存储梯度差和决策边界距离
                    grad_diff = grad_other - grad_current
                    decision_boundary_dist = (loss_other - loss_current).item()
                    
                    key = f"t{t}_k{k}"
                    grad_dict[key] = {
                        'grad_diff': grad_diff,
                        'distance': abs(decision_boundary_dist),
                        'logit_diff': decision_boundary_dist
                    }
        
        return grad_dict
    
    def _select_optimal_perturbation(self, grad_dict, overshoot):
        """选择最优扰动方向"""
        if not grad_dict:
            return torch.zeros_like(list(grad_dict.values())[0]['grad_diff'])
        
        min_perturbation_norm = float('inf')
        optimal_perturbation = None
        
        for key, info in grad_dict.items():
            grad_diff = info['grad_diff']
            distance = info['distance']
            
            # 计算扰动方向的L2范数
            grad_norm = torch.norm(grad_diff.flatten()).item()
            
            if grad_norm > 1e-8 and distance > 1e-8:
                # 计算最小扰动
                perturbation = (distance / (grad_norm ** 2)) * grad_diff
                perturbation_norm = torch.norm(perturbation.flatten()).item()
                
                if perturbation_norm < min_perturbation_norm:
                    min_perturbation_norm = perturbation_norm
                    optimal_perturbation = perturbation
        
        if optimal_perturbation is not None:
            return (1 + overshoot) * optimal_perturbation
        else:
            # 如果没有找到有效扰动，使用随机扰动
            return 0.001 * torch.randn_like(list(grad_dict.values())[0]['grad_diff'])
    
    def _adaptive_overshoot_update(self, current_norm, best_norm, current_overshoot, iteration):
        """自适应更新超调参数"""
        if iteration < 5:
            return current_overshoot
        
        # 如果扰动在减小，减少超调
        if current_norm < best_norm:
            new_overshoot = max(current_overshoot * 0.95, self.min_overshoot)
        else:
            # 如果扰动在增大，增加超调
            new_overshoot = min(current_overshoot * 1.05, self.max_overshoot)
        
        return new_overshoot
    
    def _check_convergence(self, history, window_size=5):
        """检查是否收敛"""
        if len(history) < window_size * 2:
            return False
        
        recent_values = history[-window_size:]
        previous_values = history[-window_size*2:-window_size]
        
        recent_mean = np.mean(recent_values)
        previous_mean = np.mean(previous_values)
        
        # 如果最近的变化很小，认为收敛
        relative_change = abs(recent_mean - previous_mean) / (previous_mean + 1e-8)
        return relative_change < self.convergence_threshold


def select_model():
    """交互式模型选择"""
    models = {
        1: 'parseq_tiny',
        2: 'parseq_patch16_224', 
        3: 'parseq',
        4: 'parseq_base'
    }
    
    descriptions = {
        1: '微型模型 - 最快速度，适合快速测试',
        2: '标准模型 - 平衡性能和速度',
        3: '完整模型 - 最佳性能', 
        4: '基础模型 - 标准配置'
    }
    
    print("\n请选择PARSeq模型:")
    for key, desc in descriptions.items():
        print(f"  {key}. {models[key]} - {desc}")
    
    while True:
        try:
            choice = int(input("\n请输入选择 (1-4): "))
            if choice in models:
                model_name = models[choice]
                print(f"✓ 已选择: {model_name}")
                return model_name
            else:
                print("无效选择，请输入1-4之间的数字")
        except ValueError:
            print("请输入有效数字")


def select_dataset_scope():
    """选择数据集范围"""
    scopes = {
        1: (5, "小样本测试 - 5张图像"),
        2: (15, "中等规模测试 - 15张图像"), 
        3: (30, "大规模测试 - 30张图像"),
        4: (80, "完整数据集 - 80张图像")
    }
    
    print("\n请选择测试范围:")
    for key, (num, desc) in scopes.items():
        print(f"  {key}. {desc}")
    
    while True:
        try:
            choice = int(input("\n请输入选择 (1-4): "))
            if choice in scopes:
                num_images, desc = scopes[choice]
                print(f"✓ 已选择: {desc}")
                return num_images
            else:
                print("无效选择，请输入1-4之间的数字")
        except ValueError:
            print("请输入有效数字")


def load_cute80_images(num_images=5):
    """加载CUTE80数据集图像"""
    cute80_dir = Path(__file__).parent.parent.parent / "CUTE80"
    
    if not cute80_dir.exists():
        raise FileNotFoundError(f"CUTE80数据集目录不存在: {cute80_dir}")
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(cute80_dir.glob(f"*{ext}")))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"在 {cute80_dir} 中没有找到图像文件")
    
    # 排序并选择指定数量的图像
    image_files.sort()
    selected_files = image_files[:num_images]
    
    print(f"\n从CUTE80数据集加载 {len(selected_files)} 张图像:")
    for i, img_path in enumerate(selected_files):
        print(f"  {i+1}. {img_path.name}")
    
    return selected_files


def preprocess_images(image_files, model):
    """预处理图像"""
    from strhub.data.module import SceneTextDataModule
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
    original_images = []
    image_tensors = []
    image_names = []
    
    print("\n正在预处理图像...")
    for i, img_path in enumerate(image_files):
        try:
            # 加载原始图像
            orig_img = Image.open(img_path).convert('RGB')
            original_images.append(orig_img)
            
            # 预处理
            img_tensor = transform(orig_img)
            image_tensors.append(img_tensor)
            
            # 获取图像名
            image_names.append(img_path.stem)
            
            print(f"  {i+1}. {img_path.name} -> {orig_img.size}")
            
        except Exception as e:
            print(f"  警告: 加载图像 {img_path.name} 失败: {e}")
            continue
    
    if len(image_tensors) == 0:
        raise ValueError("没有成功加载任何图像！")
    
    # 转换为batch
    images_batch = torch.stack(image_tensors).to(device)
    print(f"\n图像批次准备完成，形状: {images_batch.shape}")
    
    return original_images, images_batch, image_names


def visualize_results(original_images, original_texts, adversarial_texts, 
                     perturbations, image_names, convergence_histories, save_dir=None):
    """可视化攻击结果和收敛曲线"""
    num_images = len(original_images)
    
    # 创建子图布局：原始图像、扰动、收敛曲线
    fig = plt.figure(figsize=(20, 6 * ((num_images + 2) // 3)))
    
    rows = (num_images + 2) // 3
    cols = 3
    
    # 主标题
    fig.suptitle("SuperDeepFool攻击结果对比", fontsize=20, fontweight='bold')
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        # 创建子图网格
        ax1 = plt.subplot2grid((rows * 3, cols), (row * 3, col))
        ax2 = plt.subplot2grid((rows * 3, cols), (row * 3 + 1, col))
        ax3 = plt.subplot2grid((rows * 3, cols), (row * 3 + 2, col))
        
        # 原始图像
        ax1.imshow(original_images[i])
        ax1.set_title(f'原始图像 {i+1}\n识别: "{original_texts[i]}"', fontsize=10)
        ax1.axis('off')
        
        # 扰动可视化
        pert = perturbations[i].cpu().numpy()
        pert_vis = np.transpose(pert, (1, 2, 0))
        pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        
        ax2.imshow(pert_vis)
        attack_status = "成功" if original_texts[i] != adversarial_texts[i] else "失败"
        ax2.set_title(f'扰动可视化\n攻击后: "{adversarial_texts[i]}"\n状态: {attack_status}', fontsize=10)
        ax2.axis('off')
        
        # 收敛曲线
        if i < len(convergence_histories) and convergence_histories[i]:
            ax3.plot(convergence_histories[i], 'b-', linewidth=2)
            ax3.set_title(f'收敛曲线\n最终L2范数: {convergence_histories[i][-1]:.4f}', fontsize=10)
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('扰动L2范数')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无收敛数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('收敛曲线', fontsize=10)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "superdeepfool_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果图像已保存到: {save_path}")
    
    plt.show()


def quick_test():
    """快速测试模式"""
    print("🚀 SuperDeepFool快速测试模式")
    print("=" * 50)
    
    # 固定配置以提高速度
    config = {
        'model_name': 'parseq_tiny',
        'num_samples': 3,
        'max_iter': 10,
        'top_k_classes': 2,
        'overshoot': 0.05
    }
    
    print(f"模型: {config['model_name']}")
    print(f"样本数: {config['num_samples']}")
    print(f"最大迭代: {config['max_iter']}")
    print(f"Top-K类别: {config['top_k_classes']}")
    
    try:
        # 加载模型
        print("\n📦 加载模型...")
        start_time = time.time()
        model = load_model(config['model_name'])
        print(f"✓ 模型加载完成 ({time.time() - start_time:.1f}s)")
        
        # 加载数据
        print("\n📁 加载测试数据...")
        start_time = time.time()
        data_loader = load_cute80_data(num_samples=config['num_samples'])
        print(f"✓ 数据加载完成 ({time.time() - start_time:.1f}s)")
        
        # 初始化攻击器
        print("\n⚡ 初始化SuperDeepFool攻击器...")
        attacker = SuperDeepFoolAttacker(
            model=model,
            top_k_classes=config['top_k_classes'],
            momentum=0.8,
            adaptive_overshoot=True
        )
        
        # 执行攻击
        print(f"\n🎯 开始攻击测试...")
        attack_start = time.time()
        
        for batch_images, batch_labels in data_loader:
            adversarial_images, perturbations, iterations, histories = attacker.attack(
                batch_images, 
                max_iter=config['max_iter'],
                overshoot=config['overshoot']
            )
            
            # 计算攻击结果
            success_rate = calculate_attack_success_rate(
                model, batch_images, adversarial_images, batch_labels
            )
            
            total_time = time.time() - attack_start
            
            print(f"\n📊 快速测试结果:")
            print(f"  • 处理图像: {len(batch_images)}")
            print(f"  • 攻击成功率: {success_rate:.1%}")
            print(f"  • 平均迭代次数: {np.mean(iterations):.1f}")
            print(f"  • 总耗时: {total_time:.1f}s")
            print(f"  • 平均每张图像: {total_time/len(batch_images):.1f}s")
            
            break  # 只处理第一个批次
            
        print("\n✅ 快速测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        print("请检查环境配置和依赖项")


def main():
    """主函数"""
    print("=" * 80)
    print("SuperDeepFool对抗攻击 - 改进版DeepFool算法")
    print("适用于PARSeq文本识别模型")
    print("=" * 80)
    
    try:
        # 1. 选择模型
        model_name = select_model()
        
        # 2. 加载模型
        print(f"\n正在加载模型: {model_name}...")
        model = torch.hub.load('baudm/parseq', model_name, pretrained=True, trust_repo=True)
        model.eval()
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型加载成功!")
        print(f"  参数量: {total_params:,}")
        print(f"  输入尺寸: {model.hparams.img_size}")
        
        # 3. 选择数据集范围
        num_images = select_dataset_scope()
        
        # 4. 加载图像
        image_files = load_cute80_images(num_images)
        original_images, images_batch, image_names = preprocess_images(image_files, model)
        
        # 5. 原始预测
        print("\n正在进行原始文本识别...")
        original_texts = predict_text(model, images_batch)
        
        print("\n原始识别结果:")
        for i, (name, text) in enumerate(zip(image_names, original_texts)):
            print(f"  {i+1}. {name}: \"{text}\"")
        
        # 6. 创建SuperDeepFool攻击器
        attacker = SuperDeepFoolAttacker(model, device)
        
        # 7. 执行SuperDeepFool攻击
        print(f"\n开始执行SuperDeepFool攻击...")
        max_iter = 100
        overshoot = 0.02
        
        start_time = time.time()
        adversarial_images, perturbations, iterations, convergence_histories = attacker.superdeepfool_attack(
            images_batch, max_iter=max_iter, overshoot=overshoot
        )
        total_time = time.time() - start_time
        
        # 8. 攻击后预测
        print("\n正在识别对抗样本...")
        adversarial_texts = predict_text(model, adversarial_images)
        
        # 9. 分析结果
        print("\n" + "="*80)
        print("SuperDeepFool攻击结果分析")
        print("="*80)
        
        success_count = 0
        total_perturbation = 0
        
        for i in range(len(original_texts)):
            attack_success = original_texts[i] != adversarial_texts[i]
            if attack_success:
                success_count += 1
            
            pert_norm = torch.norm(perturbations[i]).item()
            total_perturbation += pert_norm
            
            status = "✓ 成功" if attack_success else "✗ 失败"
            print(f"{i+1}. {image_names[i]}:")
            print(f"   原始: \"{original_texts[i]}\"")
            print(f"   攻击后: \"{adversarial_texts[i]}\"")
            print(f"   状态: {status}")
            print(f"   迭代次数: {iterations[i]}")
            print(f"   扰动L2范数: {pert_norm:.6f}")
            print("-" * 60)
        
        success_rate = success_count / len(original_texts)
        avg_perturbation = total_perturbation / len(original_texts)
        avg_iterations = sum(iterations) / len(iterations)
        
        print(f"\n总体统计:")
        print(f"  攻击成功率: {success_count}/{len(original_texts)} = {success_rate:.1%}")
        print(f"  平均扰动L2范数: {avg_perturbation:.6f}")
        print(f"  平均迭代次数: {avg_iterations:.1f}")
        print(f"  总攻击时间: {total_time:.1f}秒")
        print(f"  平均每张图像: {total_time/len(original_texts):.1f}秒")
        
        # 10. 保存结果
        save_dir = Path(__file__).parent / "results"
        save_dir.mkdir(exist_ok=True)
        
        # 保存统计结果
        stats_file = save_dir / f"superdeepfool_stats_{model_name}.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("SuperDeepFool攻击统计结果\n")
            f.write("="*50 + "\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"测试图像数: {len(original_texts)}\n")
            f.write(f"攻击成功率: {success_rate:.1%}\n")
            f.write(f"平均扰动L2范数: {avg_perturbation:.6f}\n")
            f.write(f"平均迭代次数: {avg_iterations:.1f}\n")
            f.write(f"总攻击时间: {total_time:.1f}秒\n")
            f.write(f"平均每张图像: {total_time/len(original_texts):.1f}秒\n\n")
            
            for i in range(len(original_texts)):
                attack_success = original_texts[i] != adversarial_texts[i]
                status = "成功" if attack_success else "失败"
                pert_norm = torch.norm(perturbations[i]).item()
                
                f.write(f"{i+1}. {image_names[i]}:\n")
                f.write(f"   原始: \"{original_texts[i]}\"\n")
                f.write(f"   攻击后: \"{adversarial_texts[i]}\"\n")
                f.write(f"   状态: {status}\n")
                f.write(f"   迭代次数: {iterations[i]}\n")
                f.write(f"   扰动L2范数: {pert_norm:.6f}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\n详细统计结果已保存到: {stats_file}")
        
        # 11. 可视化结果
        print("\n正在生成可视化结果...")
        visualize_results(original_images, original_texts, adversarial_texts, 
                         perturbations, image_names, convergence_histories, save_dir)
        
        print("\n" + "="*80)
        print("SuperDeepFool攻击实验完成！")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("SuperDeepFool攻击脚本")
    print("1. 快速测试模式")
    print("2. 完整攻击模式")
    
    choice = input("\n请选择模式 (1/2, 默认1): ").strip()
    
    if choice == "2":
        main()
    else:
        quick_test()
