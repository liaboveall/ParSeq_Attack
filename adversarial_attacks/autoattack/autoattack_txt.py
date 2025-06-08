#!/usr/bin/env python3
"""
AutoAttack攻击算法在场景文本识别模型中的实现。
该脚本提供了各种AutoAttack配置用于评估文本识别模型的鲁棒性。
"""

import os
import sys
import string
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import warnings

# 添加父目录到路径以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strhub.models.utils import load_from_checkpoint
from strhub.data.module import SceneTextDataModule

try:
    print("尝试导入AutoAttack...")
    from autoattack import AutoAttack
    print("✅ AutoAttack导入成功！")
except ImportError as e:
    print(f"❌ AutoAttack导入失败: {e}")
    print("请先安装：")
    print("pip install git+https://github.com/fra31/auto-attack")
    sys.exit(1)
except Exception as e:
    print(f"❌ 导入AutoAttack时发生未知错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class TextRecognitionModelWrapper:
    """
    文本识别模型包装器类，用于适配AutoAttack。
    AutoAttack期望一个接收图像并返回logits的前向函数。
    """
    
    def __init__(self, model, device='cuda', charset=None):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 获取字符集
        if charset is None:
            # ParseQ默认字符集
            chars = string.digits + string.ascii_lowercase
            self.charset = ['[B]', '[E]'] + list(chars) + ['[P]']
        else:
            self.charset = charset
        self.num_classes = len(self.charset)
        
        # 测试模型输出形状
        dummy_input = torch.randn(1, 3, 32, 128).to(device)
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
            if hasattr(dummy_output, 'logits'):
                self.output_shape = dummy_output.logits.shape
            else:
                self.output_shape = dummy_output.shape
            print(f"模型输出形状: {self.output_shape}")
        
    def __call__(self, x):
        """
        AutoAttack的前向传播函数。
        
        参数:
            x: 输入图像张量，形状为(batch_size, channels, height, width)
               期望值在[0, 1]范围内
        
        返回:
            logits: 分类logits，形状为(batch_size, num_classes)
        """
        # 确保不使用梯度计算（但允许攻击时的梯度）
        if x.device != self.device:
            x = x.to(self.device)
            
        # 标准化输入到模型期望的范围 [-1, 1]
        x_normalized = (x - 0.5) / 0.5
        
        # 获取模型输出
        output = self.model(x_normalized)
        
        # 处理不同的输出格式
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
            
        # 对文本识别任务，我们需要将序列输出转换为分类问题
        # 方法1: 使用最大池化聚合所有位置的预测
        if len(logits.shape) == 3:  # (batch_size, seq_len, vocab_size)
            # 取每个位置的最大置信度，然后在序列维度上平均
            logits = torch.max(logits, dim=1)[0]  # (batch_size, vocab_size)
        
        return logits


def load_model_and_data(model_name, checkpoint_path, data_root, dataset_name):
    """加载模型和准备数据集。"""
    
    print(f"加载模型: {model_name}")
    # 从检查点加载模型
    model = load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # 直接从图像文件夹加载数据
    print(f"加载数据集: {dataset_name}")
    
    return model, None


def prepare_text_labels(images, model_wrapper, labels_text=None):
    """
    为文本识别任务准备正确的标签。
    
    参数:
        images: 输入图像张量
        model_wrapper: 模型包装器
        labels_text: 可选的文本标签列表
        
    返回:
        tensor: 标签张量，形状为 (batch_size,)
    """
    batch_size = images.shape[0]
    
    # 方法1: 基于模型的真实预测作为目标标签
    with torch.no_grad():
        model_output = model_wrapper(images)
        # 使用模型的最高置信度预测作为目标
        predicted_labels = torch.argmax(model_output, dim=1)
    
    print(f"生成了{batch_size}个样本的标签，形状: {predicted_labels.shape}")
    print(f"标签范围: {predicted_labels.min().item()} 到 {predicted_labels.max().item()}")
    return predicted_labels



def get_user_input():
    """获取用户输入的配置"""
    print("=" * 60)
    print("        文本识别模型的AutoAttack对抗攻击评估")
    print("=" * 60)
    
    # 可用模型列表
    available_models = {
        '1': ('parseq-tiny', '轻量级模型 (6M参数)'),
        '2': ('parseq', '标准模型 (23M参数)'),
        '3': ('abinet', 'ABINet模型'),
        '4': ('vitstr', 'ViTSTR模型'),
        '5': ('crnn', 'CRNN模型'),
        '6': ('trba', 'TRBA模型')
    }
    
    # 可用数据集
    available_datasets = {
        '1': 'CUTE80',
        '2': 'IIIT5K', 
        '3': 'SVT',
        '4': 'IC13_857',
        '5': 'IC15_1811'
    }
    
    # 攻击类型
    attack_configs = {
        '1': {
            'name': '标准攻击 (Linf, ε=8/255)',
            'norm': 'Linf',
            'epsilon': 8.0/255.0,
            'version': 'standard'
        },
        '2': {
            'name': '轻量攻击 (Linf, ε=4/255)',
            'norm': 'Linf', 
            'epsilon': 4.0/255.0,
            'version': 'custom',
            'attacks': ['apgd-ce', 'fab']
        },
        '3': {
            'name': 'L2攻击 (L2, ε=0.5)',
            'norm': 'L2',
            'epsilon': 0.5,
            'version': 'standard'
        }
    }
    
    print("\n📦 选择要评估的模型:")
    for key, (model_id, desc) in available_models.items():
        print(f"  {key}. {model_id} - {desc}")
    
    while True:
        model_choice = input("\n请选择模型 (1-6): ").strip()
        if model_choice in available_models:
            model_name = available_models[model_choice][0]
            break
        print("❌ 无效选择，请重新输入")
    
    print(f"\n✅ 已选择模型: {model_name}")
    
    print("\n📊 选择测试数据集:")
    for key, dataset in available_datasets.items():
        print(f"  {key}. {dataset}")
    
    while True:
        dataset_choice = input("\n请选择数据集 (1-5, 默认CUTE80): ").strip()
        if not dataset_choice:
            dataset_name = 'CUTE80'
            break
        elif dataset_choice in available_datasets:
            dataset_name = available_datasets[dataset_choice]
            break
        print("❌ 无效选择，请重新输入")
    
    print(f"✅ 已选择数据集: {dataset_name}")
    
    print("\n⚔️ 选择攻击类型:")
    for key, config in attack_configs.items():
        print(f"  {key}. {config['name']}")
    
    while True:
        attack_choice = input("\n请选择攻击类型 (1-3, 默认1): ").strip()
        if not attack_choice:
            attack_choice = '1'
        if attack_choice in attack_configs:
            attack_config = attack_configs[attack_choice]
            break
        print("❌ 无效选择，请重新输入")
    
    print(f"✅ 已选择攻击: {attack_config['name']}")
    
    # 样本数量
    while True:
        try:
            n_examples = input("\n🔢 测试样本数量 (默认10): ").strip()
            if not n_examples:
                n_examples = 10
            else:
                n_examples = int(n_examples)
            if n_examples > 0:
                break
            print("❌ 样本数量必须大于0")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    print(f"✅ 测试样本数量: {n_examples}")
    
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        use_cuda = input(f"\n💻 使用GPU加速? (y/n, 默认y): ").strip().lower()
        if use_cuda in ['n', 'no']:
            device = 'cpu'
    
    print(f"✅ 使用设备: {device}")
      # 自动检测数据根目录
    current_script_dir = Path(__file__).parent
    possible_data_roots = [
        '.',  # 当前目录
        '../..',  # 上两级目录（项目根目录）
        current_script_dir / '../..',  # 相对于脚本的项目根目录
        Path(__file__).parent.parent.parent  # 绝对路径到项目根目录
    ]
    
    data_root = '.'
    for root in possible_data_roots:
        test_path = Path(root) / dataset_name
        if test_path.exists():
            data_root = str(root)
            print(f"✅ 找到数据集路径: {test_path.absolute()}")
            break
    else:
        print(f"⚠️ 警告: 无法自动定位{dataset_name}数据集，使用默认路径")
    
    return {
        'model_name': model_name,
        'checkpoint': f'pretrained={model_name}',
        'dataset': dataset_name,
        'n_examples': n_examples,
        'device': device,
        'data_root': data_root,
        'save_dir': './results/autoattack',
        'batch_size': 4,
        **attack_config
    }


def evaluate_model_wrapper(model_wrapper, images, device):
    """
    评估模型包装器的性能，确保它正常工作。
    
    参数:
        model_wrapper: 模型包装器实例
        images: 测试图像
        device: 计算设备
    
    返回:
        bool: 是否通过基本测试
    """
    print("🔍 评估模型包装器性能...")
    
    try:
        with torch.no_grad():
            # 测试前向传播
            outputs = model_wrapper(images[:2])  # 测试前2个样本
            
            print(f"✅ 模型输出形状: {outputs.shape}")
            print(f"✅ 输出数值范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"✅ 输出均值: {outputs.mean().item():.4f}")
            print(f"✅ 输出标准差: {outputs.std().item():.4f}")
            
            # 检查输出是否合理
            if torch.isnan(outputs).any():
                print("❌ 模型输出包含NaN值")
                return False
            
            if torch.isinf(outputs).any():
                print("❌ 模型输出包含无穷值")
                return False
                
            # 检查输出是否有变化（不是全零或全相同）
            if outputs.std().item() < 1e-6:
                print("❌ 模型输出缺乏变化（可能是全零或全相同）")
                return False
            
            print("✅ 模型包装器测试通过")
            return True
            
    except Exception as e:
        print(f"❌ 模型包装器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    


def main():
    """主函数 - 交互式AutoAttack评估"""
    try:
        # 获取用户配置
        config = get_user_input()
        
        print("\n🚀 AutoAttack评估")
        print(f"模型: {config['model_name']}")
        print(f"数据集: {config['dataset']}")
        print(f"攻击: {config['name']}")
        print(f"样本数: {config['n_examples']}")
        print("-" * 40)
        
        # 设置保存路径
        save_dir = Path(config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir = save_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        config['log_path'] = str(log_dir / f"autoattack_{config['model_name']}_{config['dataset']}.log")
        
        # 创建配置对象
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        args = Config(**config)
        
        # 加载模型和数据
        model, _ = load_model_and_data(
            args.model_name, args.checkpoint, args.data_root, args.dataset
        )
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 为AutoAttack创建模型包装器
        print("🔧 创建模型包装器...")1
        model_wrapper = TextRecognitionModelWrapper(model, device)
          # 准备测试数据 - 直接从图像文件夹加载
        import glob
        from PIL import Image
        from torchvision import transforms
        
        dataset_path = os.path.join(args.data_root, args.dataset)
        print(f"🔍 正在搜索数据集路径: {dataset_path}")
        print(f"📁 绝对路径: {os.path.abspath(dataset_path)}")
        
        image_files = glob.glob(os.path.join(dataset_path, '*.jpg')) + \
                      glob.glob(os.path.join(dataset_path, '*.JPG')) + \
                      glob.glob(os.path.join(dataset_path, '*.png'))

        if not image_files:
            print(f"❌ 在{dataset_path}中未找到图像文件！")
            print(f"🔍 调试信息:")
            print(f"   - 当前工作目录: {os.getcwd()}")
            print(f"   - 脚本位置: {os.path.dirname(os.path.abspath(__file__))}")
            print(f"   - args.data_root: {args.data_root}")
            print(f"   - args.dataset: {args.dataset}")
            print(f"   - 检查路径是否存在: {os.path.exists(dataset_path)}")
            if os.path.exists(dataset_path):
                print(f"   - 目录内容: {os.listdir(dataset_path)}")
            return
        
        print(f"📸 加载{len(image_files)}张图像中的{args.n_examples}张...")
        
        # 使用模型的图像变换
        img_transform = SceneTextDataModule.get_transform((32, 128))
        
        images_list = []
        labels_list = []
        
        for i, img_path in enumerate(tqdm(image_files[:args.n_examples])):
            # 加载并预处理图像
            image = Image.open(img_path).convert('RGB')
            image_tensor = img_transform(image).unsqueeze(0)
            images_list.append(image_tensor)
            
            # 使用文件名作为伪标签（这里只是为了测试）
            filename = os.path.basename(img_path).split('.')[0]
            labels_list.append(filename)
        
        if not images_list:
            print("❌ 未找到测试数据！")
            return
        
        # 连接所有图像
        images = torch.cat(images_list, dim=0)
        
        # 移动到设备
        images = images.to(device)
        
        # 确保图像在[0,1]范围内
        images = torch.clamp(images, 0.0, 1.0)
        
        print(f"✅ 加载{len(images)}张图像 (范围: {images.min().item():.2f}-{images.max().item():.2f})")
        
        # 首先评估模型包装器
        if not evaluate_model_wrapper(model_wrapper, images, device):
            print("❌ 模型包装器评估失败，停止AutoAttack")
            return
        
        # 生成正确的标签
        print("🏷️ 生成标签...")
        labels_for_autoattack = prepare_text_labels(images, model_wrapper, labels_list)
        
        # 验证模型在干净样本上的性能
        print("🔍 验证模型性能...")
        with torch.no_grad():
            clean_outputs = model_wrapper(images)
            clean_predictions = torch.argmax(clean_outputs, dim=1)
            clean_accuracy = (clean_predictions == labels_for_autoattack).float().mean()
            print(f"📈 干净样本准确率: {clean_accuracy.item():.2%}")
            
            if clean_accuracy.item() < 0.01:  # 如果准确率太低
                print("⚠️ 警告: 干净样本准确率很低，可能存在模型或数据问题")
                print("💡 尝试使用不同的标签策略...")
                # 使用模型的top-1预测作为"真实"标签
                labels_for_autoattack = clean_predictions.clone()
                print(f"🔄 使用模型预测作为目标标签: {labels_for_autoattack[:5]}")
        
        # 运行AutoAttack评估
        print(f"\n🚀 运行AutoAttack (norm={args.norm}, ε={args.epsilon:.4f})...")
        
        # 初始化AutoAttack (关闭详细输出)
        adversary = AutoAttack(
            model_wrapper, 
            norm=args.norm, 
            eps=args.epsilon,
            version=args.version,
            verbose=False  # 关闭详细输出
        )
        
        # 如果指定了自定义攻击，则配置自定义攻击
        if args.version == 'custom' and hasattr(args, 'attacks'):
            adversary.attacks_to_run = args.attacks
            print(f"   攻击序列: {args.attacks}")
        
        # 运行AutoAttack - 不使用torch.no_grad()，因为攻击需要梯度
        adv_complete = adversary.run_standard_evaluation(
            images, labels_for_autoattack, bs=args.batch_size
        )
        
        # 保存结果
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算最终统计信息
        print("\n📊 攻击结果:")
        
        # 计算扰动统计
        if adv_complete is not None and len(adv_complete) > 0:
            perturbation = adv_complete - images
            if args.norm == 'Linf':
                max_pert = torch.norm(perturbation.view(perturbation.shape[0], -1), p=float('inf'), dim=1).max()
                avg_pert = torch.norm(perturbation.view(perturbation.shape[0], -1), p=float('inf'), dim=1).mean()
            else:
                max_pert = torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1).max()
                avg_pert = torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1).mean()
            
            # 验证对抗样本的效果
            with torch.no_grad():
                adv_outputs = model_wrapper(adv_complete)
                adv_predictions = torch.argmax(adv_outputs, dim=1)
                robust_accuracy = (adv_predictions == labels_for_autoattack).float().mean()
                
            print(f"   干净准确率:     {clean_accuracy.item():.1%}")
            print(f"   鲁棒准确率:     {robust_accuracy.item():.1%}")
            print(f"   攻击成功率:     {(1 - robust_accuracy.item()):.1%}")
            print(f"   最大扰动:       {max_pert:.4f}")
            print(f"   平均扰动:       {avg_pert:.4f}")
            print(f"   扰动预算:       {args.epsilon:.4f}")
        
        # 生成文件名
        filename = f"autoattack_{args.version}_{args.model_name}_{args.dataset}_n{len(adv_complete)}_eps{args.epsilon:.4f}.pth"
        save_path = save_dir / filename
        
        # 保存结果
        save_dict = {
            'adversarial_examples': adv_complete,
            'original_images': images,
            'labels': labels_for_autoattack,
            'config': config,
            'model_name': args.model_name,
            'dataset_name': args.dataset,
            'epsilon': args.epsilon,
            'norm': args.norm,
            'version': args.version
        }
        
        if 'max_pert' in locals():
            save_dict.update({
                'max_perturbation': max_pert.item(),
                'avg_perturbation': avg_pert.item(),
                'robust_accuracy': robust_accuracy.item(),
                'attack_success_rate': 1 - robust_accuracy.item()
            })
        
        torch.save(save_dict, save_path)
        
        print(f"\n💾 结果已保存到: {save_path}")
        print("✅ AutoAttack完成！")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断了评估过程")
    except Exception as e:
        print(f"❌ AutoAttack评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == '__main__':
    main()
