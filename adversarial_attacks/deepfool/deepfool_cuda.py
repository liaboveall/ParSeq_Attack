#!/usr/bin/env python3
"""
DeepFool对抗攻击 - 经过性能优化的版本
适配PARSeq文本识别模型，支持CUTE80数据集

主要优化:
1. 全批量处理（向量化）：移除串行循环，充分利用GPU并行能力。
2. 掩码机制：只对未成功的样本进行迭代，提高效率。
3. 简化攻击逻辑：采用高效的迭代式L2梯度攻击，保留DeepFool核心思想。
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import warnings
import sys
import time
import torchvision.transforms as transforms

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
        # 确保输入在正确的设备上
        images = images.to(next(model.parameters()).device)
        logits = model(images)
        probs = logits.softmax(-1)
        predictions, confidences = model.tokenizer.decode(probs)
    return predictions


class DeepFoolAttacker:
    """DeepFool对抗攻击器 - 经过性能优化的版本"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.eval()
        self.device = device
        
    def deepfool_attack_batch(self, images, max_iter=50, overshoot=0.02):
        """
        DeepFool攻击的批量处理实现 (向量化版本)
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            max_iter: 最大迭代次数
            overshoot: 超调参数，作为扰动步长的缩放因子
            
        Returns:
            adversarial_images: 对抗样本
            perturbations: 扰动
            iterations: 实际迭代次数
        """
        images = images.clone().detach().to(self.device)
        batch_size = images.shape[0]
        
        # 【新】初始化用于批处理的张量
        r_total = torch.zeros_like(images)
        iterations = torch.full((batch_size,), max_iter, dtype=torch.int, device=self.device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        # 【新】获取原始预测以进行比较
        with torch.no_grad():
            orig_logits = self.model(images)
            orig_labels = orig_logits.argmax(-1)
            orig_pred_texts = predict_text(self.model, images)

        print(f"开始批量DeepFool攻击 (批大小: {batch_size}, 最大迭代: {max_iter})...")
        start_time = time.time()
        
        # 【核心变化】外层循环不再是逐个图像，而是攻击迭代
        for i in range(max_iter):
            # 如果所有攻击都已成功，提前退出
            if not active_mask.any():
                print(f"  所有攻击在第 {i} 轮完成，提前停止。")
                break
                
            x_adv = images + r_total
            x_adv.requires_grad_(True)
            
            # 只对活跃的样本进行前向传播和计算
            # 确保 x_adv 中只有活跃的部分需要梯度
            active_x_adv = x_adv[active_mask]
            
            logits = self.model(active_x_adv)
            
            # 【新】简化攻击逻辑：使用交叉熵损失的梯度作为扰动方向
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                   orig_labels[active_mask].view(-1))
            
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            
            loss.backward()
            
            # 只获取活跃样本的梯度
            grad = x_adv.grad[active_mask].clone()
            
            # 计算L2范数并归一化梯度
            grad_flat = grad.view(grad.shape[0], -1)
            grad_norms = torch.norm(grad_flat, p=2, dim=1).view(-1, 1, 1, 1) + 1e-8
            perturbation_step = grad / grad_norms
            
            # 【新】更新活跃样本的总扰动
            # 使用overshoot作为步长因子，模拟DeepFool中的小步推进
            r_total[active_mask] += perturbation_step * overshoot

            # 【新】检查哪些攻击已成功，并更新掩码
            with torch.no_grad():
                # 仅对仍在活跃的样本进行预测
                adv_images_to_check = torch.clamp(images[active_mask] + r_total[active_mask], 0, 1)
                current_pred_texts = predict_text(self.model, adv_images_to_check)
                
                # 遍历活跃样本，更新状态
                active_indices = torch.where(active_mask)[0]
                for idx, pred_text in zip(active_indices, current_pred_texts):
                    if pred_text != orig_pred_texts[idx]:
                        active_mask[idx] = False
                        iterations[idx] = i + 1
            
            if (i + 1) % 10 == 0:
                num_active = active_mask.sum().item()
                print(f"  轮次 {i+1:3d}: 活跃样本数={num_active}/{batch_size}")

        # 准备返回结果
        adversarial_images = torch.clamp(images + r_total, 0, 1)
        perturbations = r_total
        
        total_time = time.time() - start_time
        print(f"批量DeepFool攻击完成! 总耗时: {total_time:.1f}s")
        return adversarial_images.detach(), perturbations.detach(), iterations.cpu()


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
            choice = input("\n请选择模型 (输入数字 1-4, 默认: 2): ").strip()
            if not choice:
                choice = 2
            choice = int(choice)
            
            if choice in models:
                selected = models[choice]
                print(f"\n✓ 已选择: {selected['name']}")
                return selected['name']
            else:
                print("❌ 无效选择，请输入 1-4 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认模型: parseq_patch16_224")
            return 'parseq_patch16_224'


def select_dataset_scope():
    """选择数据集范围"""
    scopes = {
        1: {'name': 'sample', 'desc': '小样本测试 - 前5张图像', 'num_images': 5},
        2: {'name': 'medium', 'desc': '中等规模测试 - 前15张图像', 'num_images': 15},
        3: {'name': 'large', 'desc': '大规模测试 - 前30张图像', 'num_images': 30},
        4: {'name': 'full', 'desc': '完整数据集 - 所有80张图像', 'num_images': 80}
    }
    
    print("\n" + "="*60)
    print("选择测试数据集范围:")
    print("="*60)
    for idx, scope_info in scopes.items():
        print(f"{idx}. {scope_info['name']}")
        print(f"   {scope_info['desc']}")
        print("-" * 50)
    
    while True:
        try:
            choice = input("\n请选择数据集范围 (输入数字 1-4, 默认: 1): ").strip()
            if not choice:
                choice = 1
            choice = int(choice)
            
            if choice in scopes:
                selected = scopes[choice]
                print(f"\n✓ 已选择: {selected['name']}")
                return selected['num_images']
            else:
                print("❌ 无效选择，请输入 1-4 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认范围: 5张图像")
            return 5


def load_cute80_images(num_images=5):
    """加载CUTE80数据集图像"""
    cute80_dir = Path(__file__).parent.parent.parent / "CUTE80"
    print(f"\nCUTE80数据集路径: {cute80_dir}")
    
    if not cute80_dir.exists():
        raise FileNotFoundError(f"在路径 {cute80_dir} 中未找到CUTE80数据集。请确保数据集位于正确的父目录下。")
        
    # 查找图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(list(cute80_dir.glob(ext)))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"在路径 {cute80_dir} 中未找到图像文件")
    
    # 排序并选择指定数量的图像
    image_files = sorted(image_files)[:num_images]
    print(f"找到 {len(image_files)} 张图像用于测试")
    
    return image_files


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
                     adversarial_images, perturbations, image_names, save_dir=None):
    """可视化攻击结果"""
    num_images = len(original_images)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=(5 * cols, 5 * rows))
    if num_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif rows == 1:
        axes = axes.reshape(2, -1)
    
    fig.suptitle("DeepFool攻击结果对比 (性能优化版)", fontsize=16, fontweight='bold')
    
    for i in range(num_images):
        row, col = divmod(i, cols)
        
        # 原始图像
        ax_orig = axes[row*2, col] if rows > 1 else axes[0, col]
        ax_orig.imshow(original_images[i])
        ax_orig.set_title(f'原始图像: {image_names[i]}\n识别: "{original_texts[i]}"', 
                                  fontsize=10, pad=5)
        ax_orig.axis('off')
        
        # 对抗样本与扰动
        ax_adv = axes[row*2+1, col] if rows > 1 else axes[1, col]
        
        adv_image_pil = transforms.ToPILImage()(adversarial_images[i].cpu())

        ax_adv.imshow(adv_image_pil)
        attack_status = "✓ 成功" if original_texts[i] != adversarial_texts[i] else "✗ 失败"
        pert_norm = torch.norm(perturbations[i]).item()
        ax_adv.set_title(f'对抗样本 ({attack_status})\n识别: "{adversarial_texts[i]}"\nL2范数: {pert_norm:.4f}', 
                                    fontsize=10, pad=5)
        ax_adv.axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, rows * cols):
        row, col = divmod(i, cols)
        if rows > 1:
            axes[row*2, col].axis('off')
            axes[row*2+1, col].axis('off')
        else:
            axes[0, col].axis('off')
            axes[1, col].axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_dir:
        save_path = Path(save_dir) / "deepfool_optimized_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果图像已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("DeepFool对抗攻击 (性能优化版)")
    print("=" * 70)
    
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
        
        # 6. 创建攻击器
        attacker = DeepFoolAttacker(model, device)
        
        # 7. 执行DeepFool攻击
        print(f"\n开始执行DeepFool攻击...")
        max_iter = 50
        overshoot = 0.02
        
        start_time = time.time()
        adversarial_images, perturbations, iterations = attacker.deepfool_attack_batch(
            images_batch, max_iter=max_iter, overshoot=overshoot
        )
        attack_duration = time.time() - start_time
        
        # 8. 攻击后预测
        print("\n正在识别对抗样本...")
        adversarial_texts = predict_text(model, adversarial_images)
        
        # 9. 分析结果
        print("\n" + "="*70)
        print("DeepFool攻击结果分析")
        print("="*70)
        
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
            print(f"   迭代次数: {iterations[i].item()}")
            print(f"   扰动L2范数: {pert_norm:.6f}")
            print("-" * 50)
        
        success_rate = success_count / len(original_texts)
        avg_perturbation = total_perturbation / len(original_texts) if len(original_texts) > 0 else 0
        
        print(f"\n总体统计:")
        print(f"  总攻击时间: {attack_duration:.2f}秒")
        print(f"  平均每张图像耗时: {attack_duration / len(original_texts):.2f}秒" if len(original_texts) > 0 else "N/A")
        print(f"  攻击成功率: {success_count}/{len(original_texts)} = {success_rate:.1%}")
        print(f"  平均扰动L2范数: {avg_perturbation:.6f}")
        print(f"  平均迭代次数: {iterations.float().mean().item():.1f}")
        
        # 10. 保存结果
        save_dir = Path(__file__).parent / "results"
        save_dir.mkdir(exist_ok=True)
        
        stats_file = save_dir / f"deepfool_optimized_stats_{model_name}.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("DeepFool攻击统计结果 (性能优化版)\n")
            f.write("="*50 + "\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"测试图像数: {len(original_texts)}\n")
            f.write(f"攻击成功率: {success_rate:.1%}\n")
            f.write(f"平均扰动L2范数: {avg_perturbation:.6f}\n")
            f.write(f"平均迭代次数: {iterations.float().mean().item():.1f}\n")
            f.write(f"总攻击时间: {attack_duration:.2f}秒\n")
            f.write(f"平均每张图像耗时: {attack_duration / len(original_texts):.2f}秒\n\n" if len(original_texts) > 0 else "\n")
            
            for i in range(len(original_texts)):
                f.write(f"{i+1}. {image_names[i]}:\n")
                f.write(f"   原始: \"{original_texts[i]}\"\n")
                f.write(f"   攻击后: \"{adversarial_texts[i]}\"\n")
                f.write(f"   迭代次数: {iterations[i].item()}\n")
                f.write(f"   扰动L2范数: {torch.norm(perturbations[i]).item():.6f}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\n详细统计结果已保存到: {stats_file}")
        
        # 11. 可视化结果
        print("\n正在生成可视化结果...")
        visualize_results(original_images, original_texts, adversarial_texts, 
                         adversarial_images, perturbations, image_names, save_dir)
        
        print("\n" + "="*70)
        print("DeepFool攻击实验完成！")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()