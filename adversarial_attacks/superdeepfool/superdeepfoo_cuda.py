#!/usr/bin/env python3
"""
内存优化与向量化
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
    """SuperDeepFool对抗攻击器 - 优化版DeepFool算法"""
    
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
        
    def attack(self, images, max_iter=100, overshoot=0.02, chunk_size=None):
        images = images.clone().detach().to(self.device)
        batch_size = images.shape[0]
        
        # 自动确定最优的分块大小
        if chunk_size is None:
            # 基于GPU内存和批次大小确定分块大小
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory = total_memory - torch.cuda.memory_allocated()
                # 保守估计，使用60%的可用内存
                safe_memory = free_memory * 0.6
                # 估算单个样本的内存使用量（包括梯度）
                sample_memory = images[0:1].numel() * 4 * 10  # 4 bytes per float32, 10倍安全系数
                chunk_size = max(1, min(batch_size, int(safe_memory // sample_memory)))
            else:
                chunk_size = min(4, batch_size)  # CPU情况下使用较小的分块
        
        print(f"开始优化SuperDeepFool攻击 (批大小: {batch_size}, 分块: {chunk_size}, 最大迭代: {max_iter})...")
        
        # 初始化结果容器
        adversarial_images = torch.zeros_like(images)
        r_total_all = torch.zeros_like(images)
        iterations_all = []
        convergence_histories_all = []
        
        start_time = time.time()
        
        # 分块处理
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk_images = images[chunk_start:chunk_end]
            chunk_batch_size = chunk_images.shape[0]
            
            print(f"  处理分块 {chunk_start//chunk_size + 1}/{(batch_size + chunk_size - 1)//chunk_size} "
                  f"(样本 {chunk_start+1}-{chunk_end})...")
            
            # 对当前分块执行攻击
            chunk_adv, chunk_pert, chunk_iter, chunk_hist = self._attack_chunk_optimized(
                chunk_images, max_iter, overshoot
            )
            
            # 存储结果
            adversarial_images[chunk_start:chunk_end] = chunk_adv
            r_total_all[chunk_start:chunk_end] = chunk_pert
            iterations_all.extend(chunk_iter)
            convergence_histories_all.extend(chunk_hist)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        
        print(f"\n优化SuperDeepFool攻击完成!")
        print(f"  处理样本: {batch_size}")
        print(f"  总耗时: {total_time:.1f}s, 平均每张: {total_time/batch_size:.2f}s")
        
        return adversarial_images.detach(), r_total_all.detach(), iterations_all, convergence_histories_all
    
    def _attack_chunk_optimized(self, chunk_images, max_iter, overshoot):
        """对单个分块执行优化攻击"""
        chunk_size = chunk_images.shape[0]
        
        # 初始化分块变量
        x_chunk = chunk_images.clone()
        r_total_chunk = torch.zeros_like(x_chunk)
        
        # 获取原始预测（批量）
        with torch.no_grad():
            orig_logits_chunk = self.model(x_chunk)
            orig_pred_texts = predict_text(self.model, x_chunk)
        
        # 分块SuperDeepFool变量
        momentum_buffer_chunk = torch.zeros_like(x_chunk)
        current_overshoot_chunk = torch.full((chunk_size,), overshoot, device=self.device)
        convergence_histories = [[] for _ in range(chunk_size)]
        iterations_per_sample = torch.zeros(chunk_size, dtype=torch.int, device=self.device)
        
        # 追踪攻击成功状态
        attack_success_mask = torch.zeros(chunk_size, dtype=torch.bool, device=self.device)
        
        # 预计算常用数据以减少重复计算
        check_interval = 5  # 检查间隔
        progress_interval = 20  # 进度报告间隔
        
        for i in range(max_iter):
            # 只处理尚未成功的样本
            active_mask = ~attack_success_mask
            if not active_mask.any():
                break
            
            active_indices = torch.where(active_mask)[0]
            
            # 使用torch.no_grad()上下文来减少内存使用
            x_active = x_chunk[active_mask]
            r_active = r_total_chunk[active_mask]
            x_adv_active = x_active + r_active
            x_adv_active.requires_grad_(True)  # 确保设置梯度
            
            # 前向传播
            logits_active = self.model(x_adv_active)
            
            # 定期检查预测变化
            if i % check_interval == 0:
                with torch.no_grad():
                    current_pred_texts = predict_text(self.model, x_adv_active)
                    
                    for j, active_idx in enumerate(active_indices):
                        if current_pred_texts[j] != orig_pred_texts[active_idx]:
                            if not attack_success_mask[active_idx]:
                                attack_success_mask[active_idx] = True
                                iterations_per_sample[active_idx] = i + 1
            
            # 高效梯度计算（简化版）
            perturbations = self._compute_efficient_perturbations(x_adv_active, logits_active, 
                                                                current_overshoot_chunk[active_mask])
            
            # 批量应用动量和更新
            momentum_buffer_active = momentum_buffer_chunk[active_mask]
            momentum_buffer_active = (self.momentum * momentum_buffer_active + 
                                    (1 - self.momentum) * perturbations)
            momentum_buffer_chunk[active_mask] = momentum_buffer_active
            r_total_chunk[active_mask] = r_total_chunk[active_mask] + momentum_buffer_active
            
            # 更新收敛历史（降低频率）
            if i % 2 == 0:  # 每2轮更新一次
                current_norms = torch.norm(r_total_chunk[active_mask].view(len(active_indices), -1), dim=1)
                for j, active_idx in enumerate(active_indices):
                    convergence_histories[active_idx].append(current_norms[j].item())
            
            # 进度显示
            if i % progress_interval == 0 and len(active_indices) > 0:
                active_count = len(active_indices)
                success_count = attack_success_mask.sum().item()
                print(f"      轮次 {i+1:3d}: 活跃={active_count}, 成功={success_count}")
        
        # 设置未成功样本的迭代次数
        iterations_per_sample[~attack_success_mask] = max_iter
        
        # 生成最终对抗样本
        adversarial_chunk = torch.clamp(x_chunk + r_total_chunk, 0, 1)
        
        return (adversarial_chunk, r_total_chunk, 
                iterations_per_sample.cpu().tolist(), convergence_histories)
    
    def _compute_efficient_perturbations(self, x_adv_batch, logits_batch, overshoot_batch):
        """计算高效扰动 - 最简化版本"""
        batch_size = x_adv_batch.shape[0]
        seq_len, vocab_size = logits_batch.shape[1], logits_batch.shape[2]
        
        # 只处理第一个时间步的决策边界（最重要）
        first_step_logits = logits_batch[:, 0, :]  # [B, V]
        
        # 获取top-2类别
        top2_values, top2_indices = torch.topk(first_step_logits, 2, dim=-1)
        
        perturbations = []
        
        for i in range(batch_size):
            x_sample = x_adv_batch[i:i+1].clone()  # 确保复制
            x_sample.requires_grad_(True)  # 重新设置requires_grad
            
            # 重新计算logits以确保计算图正确
            sample_logits = self.model(x_sample)
            first_step_sample_logits = sample_logits[:, 0, :]
            
            class1_logit = first_step_sample_logits[0, top2_indices[i, 0]]
            class2_logit = first_step_sample_logits[0, top2_indices[i, 1]]
            
            # 计算梯度
            self.model.zero_grad()
            if x_sample.grad is not None:
                x_sample.grad.zero_()
            
            # 简化的梯度计算
            loss_diff = class2_logit - class1_logit
            loss_diff.backward(retain_graph=True)
            
            grad = x_sample.grad.clone() if x_sample.grad is not None else torch.zeros_like(x_sample)
            
            # 应用超调参数，去掉额外的维度
            perturbation = (grad * overshoot_batch[i].item() * 0.01).squeeze(0)  # 移除批次维度
            perturbations.append(perturbation)
        
        return torch.stack(perturbations) if perturbations else torch.zeros_like(x_adv_batch)


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
        adversarial_images, perturbations, iterations, convergence_histories = attacker.attack(
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
    main()