#!/usr/bin/env python3
"""
DeepFool对抗攻击

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


class DeepFoolAttacker:
    """DeepFool对抗攻击器 - 专为PARSeq模型设计"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.eval()
        self.device = device
        
    def deepfool_attack(self, images, max_iter=50, overshoot=0.02):
        """
        DeepFool攻击实现
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            max_iter: 最大迭代次数
            overshoot: 超调参数，增加扰动的幅度
            
        Returns:
            adversarial_images: 对抗样本
            perturbations: 扰动
            iterations: 实际迭代次数
        """
        images = images.clone().detach().to(self.device)
        batch_size = images.shape[0]
        
        adversarial_images = images.clone()
        perturbations = torch.zeros_like(images)
        iterations = torch.zeros(batch_size, dtype=torch.int)
        
        print(f"开始DeepFool攻击 (批大小: {batch_size}, 最大迭代: {max_iter})...")
        
        for batch_idx in range(batch_size):
            print(f"  处理图像 {batch_idx + 1}/{batch_size}...", end="")
            
            # 单个图像处理
            x = images[batch_idx:batch_idx+1].clone()
            x.requires_grad_(True)
            
            # 获取原始预测
            with torch.no_grad():
                orig_logits = self.model(x)
                orig_probs = orig_logits.softmax(-1)
                orig_pred = orig_logits.argmax(-1)
            
            # 初始化
            r_total = torch.zeros_like(x)
            
            for i in range(max_iter):
                # 前向传播
                x_adv = x + r_total
                x_adv.requires_grad_(True)
                
                logits = self.model(x_adv)
                current_pred = logits.argmax(-1)
                
                # 检查是否已经改变预测
                if not torch.equal(current_pred, orig_pred):
                    iterations[batch_idx] = i
                    break
                
                # 计算梯度
                self.model.zero_grad()
                
                # 对每个时间步和字符维度处理
                seq_len, vocab_size = logits.shape[1], logits.shape[2]
                
                # 找到当前预测的类别和概率最高的其他类别
                current_logits = logits.view(-1, vocab_size)  # [seq_len, vocab_size]
                current_preds = current_logits.argmax(-1)
                
                min_perturbation = float('inf')
                optimal_r = torch.zeros_like(x)
                
                # 对每个位置计算最小扰动
                for t in range(seq_len):
                    current_class = current_preds[t]
                    logit_t = current_logits[t]
                    
                    # 计算当前类别的损失
                    loss_current = logit_t[current_class]
                    loss_current.backward(retain_graph=True)
                    grad_current = x_adv.grad.clone() if x_adv.grad is not None else torch.zeros_like(x_adv)
                    x_adv.grad.zero_() if x_adv.grad is not None else None
                    
                    # 找到第二高的类别
                    logit_t_sorted, indices = torch.sort(logit_t, descending=True)
                    second_class = indices[1] if indices[0] == current_class else indices[0]
                    
                    # 计算第二高类别的损失
                    loss_second = logit_t[second_class]
                    loss_second.backward(retain_graph=True)
                    grad_second = x_adv.grad.clone() if x_adv.grad is not None else torch.zeros_like(x_adv)
                    x_adv.grad.zero_() if x_adv.grad is not None else None
                    
                    # 计算决策边界方向的梯度
                    w = grad_second - grad_current
                    f = (loss_second - loss_current).item()
                    
                    # 计算最小扰动
                    w_norm = torch.norm(w.flatten()).item()
                    if w_norm > 1e-8:
                        r_i = abs(f) / (w_norm ** 2) * w
                        r_i_norm = torch.norm(r_i.flatten()).item()
                        
                        if r_i_norm < min_perturbation:
                            min_perturbation = r_i_norm
                            optimal_r = r_i
                
                # 更新总扰动
                if min_perturbation < float('inf'):
                    r_total = r_total + (1 + overshoot) * optimal_r
                else:
                    # 如果无法找到最小扰动，使用随机扰动
                    r_total = r_total + 0.01 * torch.randn_like(x)
                
            else:
                # 达到最大迭代次数
                iterations[batch_idx] = max_iter
            
            # 存储结果
            adversarial_images[batch_idx] = torch.clamp(x + r_total, 0, 1).squeeze(0)
            perturbations[batch_idx] = r_total.squeeze(0)
            
            print(f" 完成 (迭代: {iterations[batch_idx].item()})")
        
        print("DeepFool攻击完成!")
        return adversarial_images.detach(), perturbations.detach(), iterations


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
            choice = input("\n请选择数据集范围 (输入数字 1-4): ").strip()
            choice = int(choice)
            
            if choice in scopes:
                selected = scopes[choice]
                print(f"\n✓ 已选择: {selected['name']}")
                print(f"  {selected['desc']}")
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
                     perturbations, image_names, save_dir=None):
    """可视化攻击结果"""
    num_images = len(original_images)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=(5 * cols, 4 * rows))
    if num_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif rows == 1:
        axes = axes.reshape(2, -1)
    
    fig.suptitle("DeepFool攻击结果对比", fontsize=16, fontweight='bold')
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        # 原始图像
        axes[row*2, col].imshow(original_images[i])
        axes[row*2, col].set_title(f'原始图像 {i+1}\n识别: "{original_texts[i]}"', 
                                  fontsize=10, pad=5)
        axes[row*2, col].axis('off')
        
        # 扰动可视化
        pert = perturbations[i].cpu().numpy()
        pert_vis = np.transpose(pert, (1, 2, 0))
        pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        
        axes[row*2+1, col].imshow(pert_vis)
        attack_status = "成功" if original_texts[i] != adversarial_texts[i] else "失败"
        axes[row*2+1, col].set_title(f'扰动可视化\n攻击后: "{adversarial_texts[i]}"\n状态: {attack_status}', 
                                    fontsize=10, pad=5)
        axes[row*2+1, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row*2, col].axis('off')
        axes[row*2+1, col].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "deepfool_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果图像已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("DeepFool对抗攻击 - PARSeq文本识别模型")
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
        
        adversarial_images, perturbations, iterations = attacker.deepfool_attack(
            images_batch, max_iter=max_iter, overshoot=overshoot
        )
        
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
        avg_perturbation = total_perturbation / len(original_texts)
        
        print(f"\n总体统计:")
        print(f"  攻击成功率: {success_count}/{len(original_texts)} = {success_rate:.1%}")
        print(f"  平均扰动L2范数: {avg_perturbation:.6f}")
        print(f"  平均迭代次数: {iterations.float().mean().item():.1f}")
        
        # 10. 保存结果
        save_dir = Path(__file__).parent / "results"
        save_dir.mkdir(exist_ok=True)
        
        # 保存统计结果
        stats_file = save_dir / f"deepfool_stats_{model_name}.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("DeepFool攻击统计结果\n")
            f.write("="*50 + "\n")
            f.write(f"模型: {model_name}\n")
            f.write(f"测试图像数: {len(original_texts)}\n")
            f.write(f"攻击成功率: {success_rate:.1%}\n")
            f.write(f"平均扰动L2范数: {avg_perturbation:.6f}\n")
            f.write(f"平均迭代次数: {iterations.float().mean().item():.1f}\n\n")
            
            for i in range(len(original_texts)):
                attack_success = original_texts[i] != adversarial_texts[i]
                status = "成功" if attack_success else "失败"
                pert_norm = torch.norm(perturbations[i]).item()
                
                f.write(f"{i+1}. {image_names[i]}:\n")
                f.write(f"   原始: \"{original_texts[i]}\"\n")
                f.write(f"   攻击后: \"{adversarial_texts[i]}\"\n")
                f.write(f"   状态: {status}\n")
                f.write(f"   迭代次数: {iterations[i].item()}\n")
                f.write(f"   扰动L2范数: {pert_norm:.6f}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\n详细统计结果已保存到: {stats_file}")
        
        # 11. 可视化结果
        print("\n正在生成可视化结果...")
        visualize_results(original_images, original_texts, adversarial_texts, 
                         perturbations, image_names, save_dir)
        
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
