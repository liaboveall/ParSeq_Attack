#!/usr/bin/env python3
"""
AdvGAN对抗攻击脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from pathlib import Path
import warnings
import time

# 添加父目录到Python路径以导入strhub
sys.path.append(str(Path(__file__).parent.parent.parent))

# 导入PARSeq相关模块
try:
    import torch.hub
    from strhub.data.module import SceneTextDataModule
    from strhub.data.utils import Tokenizer
except ImportError as e:
    print(f"导入strhub模块失败: {e}")
    print("请确保正确安装了strhub包并且路径设置正确")
    sys.exit(1)

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

warnings.filterwarnings('ignore')

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def predict_text(model, images):
    """使用模型的tokenizer正确解码文本，与deepfool/interactive风格一致"""
    with torch.no_grad():
        logits = model(images)
        probs = logits.softmax(-1)
        predictions, confidences = model.tokenizer.decode(probs)
    return predictions


def load_model(model_name='parseq'):
    """加载PARSeq模型，与其他攻击脚本风格一致"""
    print(f"正在加载模型: {model_name}...")
    try:
        model = torch.hub.load('baudm/parseq', model_name, pretrained=True, trust_repo=True)
        model = model.eval().to(device)
        print(f"✓ 模型 {model_name} 加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def load_cute80_images(use_all=False, num_images=5):
    """加载CUTE80数据集图像，与interactive攻击保持一致的接口"""
    cute80_dir = Path("CUTE80")
    if not cute80_dir.exists():
        cute80_dir = Path(__file__).parent.parent.parent / "CUTE80"
    
    if not cute80_dir.exists():
        raise FileNotFoundError(f"找不到CUTE80数据集目录: {cute80_dir}")
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
        image_files.extend(cute80_dir.glob(ext))
    
    image_files = sorted(image_files)[:num_images if not use_all else None]
    
    if not image_files:
        raise FileNotFoundError("CUTE80目录中没有找到任何图像文件")
    
    print(f"正在加载 {len(image_files)} 张CUTE80图像...")
    return preprocess_images(image_files)


def preprocess_images(image_paths):
    """预处理图像，与其他攻击脚本风格一致"""
    from strhub.data.module import SceneTextDataModule
    transform = SceneTextDataModule.get_transform((32, 128))
    
    all_original_images = []
    all_image_tensors = []
    all_image_names = []
    
    for i, img_path in enumerate(image_paths):
        try:
            print(f"  处理图像 {i+1}/{len(image_paths)}: {img_path.name}")
            
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


class AdvGANGenerator(nn.Module):
    """AdvGAN生成器网络 - 优化版本"""
    
    def __init__(self, input_channels=3):
        super(AdvGANGenerator, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            # 第一层: 32x128 -> 16x64
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层: 16x64 -> 8x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层: 8x32 -> 4x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第四层: 4x16 -> 2x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            # 第一层: 2x8 -> 4x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第二层: 4x16 -> 8x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三层: 8x32 -> 16x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第四层: 16x64 -> 32x128
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        perturbation = self.decoder(encoded)
        return perturbation


class AdvGANDiscriminator(nn.Module):
    """AdvGAN判别器网络 - 优化版本"""
    
    def __init__(self, input_channels=3):
        super(AdvGANDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 第一层: 32x128 -> 16x64
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层: 16x64 -> 8x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层: 8x32 -> 4x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第四层: 4x16 -> 2x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最后一层: 2x8 -> 1x1
            nn.Conv2d(512, 1, kernel_size=(2, 8), stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


class AdvGANAttacker:
    """AdvGAN攻击器 - 与deepfool/interactive风格一致"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.eval()
        self.device = device
        
        # 初始化生成器和判别器
        self.generator = AdvGANGenerator().to(device)
        self.discriminator = AdvGANDiscriminator().to(device)
        
        # 优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 损失函数权重
        self.adv_weight = 1.0
        self.gan_weight = 1.0
        self.hinge_weight = 10.0
        
    def advgan_attack(self, images, epochs=100, target_success_rate=0.8):
        """
        执行AdvGAN攻击，风格与deepfool/interactive一致
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            epochs: 训练轮数
            target_success_rate: 目标成功率
            
        Returns:
            adversarial_images: 对抗样本
            attack_success: 攻击成功标志
            final_perturbations: 最终扰动
        """
        images = images.clone().detach().to(self.device)
        batch_size = images.shape[0]
        
        print(f"开始AdvGAN攻击 (批大小: {batch_size}, 训练轮数: {epochs})...")
        
        # 获取原始预测
        with torch.no_grad():
            original_texts = predict_text(self.model, images)
        
        print(f"原始预测: {original_texts}")
        
        # 训练循环
        best_adv_images = images.clone()
        best_success_rate = 0.0
        
        for epoch in range(epochs):
            # 生成对抗扰动
            perturbations = self.generator(images)
            perturbations = torch.clamp(perturbations, -0.1, 0.1)  # 限制扰动大小
            
            # 生成对抗样本
            adv_images = torch.clamp(images + perturbations, 0, 1)
            
            # ================
            # 训练判别器
            # ================
            self.d_optimizer.zero_grad()
            
            # 真实图像
            real_outputs = self.discriminator(images)
            d_loss_real = F.binary_cross_entropy(real_outputs, torch.ones_like(real_outputs))
            
            # 对抗样本
            fake_outputs = self.discriminator(adv_images.detach())
            d_loss_fake = F.binary_cross_entropy(fake_outputs, torch.zeros_like(fake_outputs))
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()
            
            # ================
            # 训练生成器
            # ================
            self.g_optimizer.zero_grad()
            
            # 对抗损失 - 让对抗样本预测结果与原始不同
            adv_logits = self.model(adv_images)
            
            # 计算对抗损失 - 最大化与原始预测的差异
            with torch.no_grad():
                orig_logits = self.model(images)
            
            adv_loss = -F.kl_div(F.log_softmax(adv_logits, dim=-1), 
                                F.softmax(orig_logits, dim=-1), 
                                reduction='batchmean')
            
            # GAN损失 - 让判别器认为对抗样本是真实的
            fake_outputs = self.discriminator(adv_images)
            gan_loss = F.binary_cross_entropy(fake_outputs, torch.ones_like(fake_outputs))
            
            # Hinge损失 - 限制扰动大小
            hinge_loss = torch.mean(torch.max(torch.zeros_like(perturbations), 
                                            torch.abs(perturbations) - 0.05))
            
            # 总损失
            g_loss = (self.adv_weight * adv_loss + 
                     self.gan_weight * gan_loss + 
                     self.hinge_weight * hinge_loss)
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # 评估当前攻击效果
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    current_texts = predict_text(self.model, adv_images)
                    success_count = sum(1 for orig, curr in zip(original_texts, current_texts) 
                                      if orig != curr)
                    success_rate = success_count / len(original_texts)
                    
                    print(f"  轮次 {epoch+1:3d}/{epochs}: "
                          f"D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}, "
                          f"成功率={success_rate:.1%} ({success_count}/{len(original_texts)})")
                    
                    # 保存最佳结果
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_adv_images = adv_images.clone()
                    
                    # 提前停止条件
                    if success_rate >= target_success_rate:
                        print(f"  达到目标成功率 {target_success_rate:.1%}，提前停止训练")
                        break
        
        # 计算最终结果
        with torch.no_grad():
            final_texts = predict_text(self.model, best_adv_images)
            final_perturbations = best_adv_images - images
            attack_success = [orig != final for orig, final in zip(original_texts, final_texts)]
        
        print(f"AdvGAN攻击完成！最终成功率: {best_success_rate:.1%}")
        return best_adv_images.detach(), attack_success, final_perturbations.detach()


def batch_attack_and_predict(attacker, images, **kwargs):
    """批量执行AdvGAN攻击，与interactive风格一致"""
    return attacker.advgan_attack(images, **kwargs)


def visualize_attack_results(original_images, adversarial_images, original_texts, 
                           adversarial_texts, attack_success, image_names, 
                           perturbations=None, max_display=10):
    """可视化攻击结果，与其他攻击脚本风格一致"""
    num_images = min(len(original_images), max_display)
    
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # 原始图像
        if isinstance(original_images[i], torch.Tensor):
            orig_img = tensor_to_image(original_images[i])
        else:
            orig_img = original_images[i]
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'原始图像\n"{original_texts[i]}"', fontsize=10)
        axes[i, 0].axis('off')
        
        # 对抗样本
        if isinstance(adversarial_images[i], torch.Tensor):
            adv_img = tensor_to_image(adversarial_images[i])
        else:
            adv_img = adversarial_images[i]
        axes[i, 1].imshow(adv_img)
        status = "✓" if attack_success[i] else "✗"
        axes[i, 1].set_title(f'对抗样本 {status}\n"{adversarial_texts[i]}"', fontsize=10)
        axes[i, 1].axis('off')
        
        # 扰动可视化
        if perturbations is not None:
            pert_img = tensor_to_image(torch.clamp(perturbations[i] * 10 + 0.5, 0, 1))
            axes[i, 2].imshow(pert_img)
            axes[i, 2].set_title('扰动 (放大10倍)', fontsize=10)
            axes[i, 2].axis('off')
        else:
            axes[i, 2].axis('off')
        
        # 差异图
        if isinstance(original_images[i], torch.Tensor) and isinstance(adversarial_images[i], torch.Tensor):
            diff = torch.abs(adversarial_images[i] - original_images[i])
            diff_img = tensor_to_image(torch.clamp(diff * 5, 0, 1))
            axes[i, 3].imshow(diff_img)
            axes[i, 3].set_title('差异 (放大5倍)', fontsize=10)
            axes[i, 3].axis('off')
        else:
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()


def tensor_to_image(tensor):
    """将tensor转换为PIL图像进行显示"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().numpy()


def print_attack_summary(original_texts, adversarial_texts, attack_success, image_names):
    """打印攻击总结，与interactive风格一致"""
    print("\n" + "="*80)
    print("AdvGAN攻击结果总结")
    print("="*80)
    
    total_images = len(original_texts)
    success_count = sum(attack_success)
    success_rate = success_count / total_images
    
    print(f"总图像数量: {total_images}")
    print(f"攻击成功率: {success_count}/{total_images} = {success_rate:.1%}")
    print(f"攻击失败率: {total_images - success_count}/{total_images} = {1-success_rate:.1%}")
    
    print(f"\n详细结果 (显示前10个):")
    for i, (orig, adv, success, name) in enumerate(zip(original_texts[:10], 
                                                       adversarial_texts[:10], 
                                                       attack_success[:10], 
                                                       image_names[:10])):
        status = "✓" if success else "✗"
        print(f"  {i+1:2d}. {name}: \"{orig}\" -> \"{adv}\" {status}")
    
    if total_images > 10:
        print(f"  ... (还有 {total_images - 10} 个结果)")


def select_model():
    """交互式模型选择，与其他攻击脚本保持一致"""
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
    """选择数据集范围，与interactive保持一致"""
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
                return selected['use_all'], selected['num_images']
            else:
                print("❌ 无效选择，请输入 1-4 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认设置: 小样本测试")
            return False, 5


def main():
    """主函数 - 执行完整的AdvGAN攻击流程"""
    print("="*80)
    print("PARSeq AdvGAN 对抗攻击程序")
    print("="*80)
    
    try:
        # 1. 交互式选择模型
        model_name = select_model()
        
        # 2. 选择数据集范围
        use_all, num_images = select_dataset_scope()
        
        # 3. 加载模型
        model = load_model(model_name)
        
        # 4. 加载数据
        original_images, image_tensors, image_names = load_cute80_images(use_all, num_images)
        
        # 5. 获取原始预测
        print("\n正在获取原始预测...")
        images_batch = torch.stack(image_tensors).to(device)
        original_texts = predict_text(model, images_batch)
        
        print("原始预测结果:")
        for i, (name, text) in enumerate(zip(image_names, original_texts)):
            print(f"  {i+1:2d}. {name}: \"{text}\"")
        
        # 6. 初始化攻击器
        print(f"\n初始化AdvGAN攻击器...")
        attacker = AdvGANAttacker(model, device)
        
        # 7. 执行攻击
        print(f"\n开始AdvGAN攻击...")
        start_time = time.time()
        
        adversarial_images, attack_success, perturbations = batch_attack_and_predict(
            attacker, 
            images_batch, 
            epochs=100,
            target_success_rate=0.8
        )
        
        attack_time = time.time() - start_time
        print(f"攻击完成，用时: {attack_time:.2f}秒")
        
        # 8. 获取对抗样本预测
        adversarial_texts = predict_text(model, adversarial_images)
        
        # 9. 打印结果总结
        print_attack_summary(original_texts, adversarial_texts, attack_success, image_names)
        
        # 10. 可视化结果
        print(f"\n显示攻击结果可视化...")
        visualize_attack_results(
            original_images,
            adversarial_images, 
            original_texts,
            adversarial_texts,
            attack_success,
            image_names,
            perturbations,
            max_display=min(10, len(original_images))
        )
        
        print("程序执行完成！")
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 只保留交互式模式
    main()
