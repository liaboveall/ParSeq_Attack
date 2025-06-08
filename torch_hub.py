#!/usr/bin/env python3
"""
支持的模型：
- parseq_tiny: 轻量级模型 (6M 参数)
- parseq: 基础模型 (23M 参数) 
- parseq_patch16_224: 高分辨率模型
- abinet: ABINet 模型
- vitstr: ViTSTR 模型
- crnn: CRNN 模型
- trba: TRBA 模型
"""

import torch
from PIL import Image
import argparse
import os


def list_available_models():
    """列出所有可用的预训练模型"""
    print("🚀 可用的预训练模型：")
    models = torch.hub.list('baudm/parseq', trust_repo=True)
    models = [m for m in models if not m.startswith('create_')]
    
    for model in models:
        print(f"  - {model}")
    return models


def load_model(model_name='parseq_tiny', device='cpu'):
    """加载预训练模型"""
    print(f"📥 加载 {model_name} 预训练模型...")
    
    # 加载模型
    model = torch.hub.load('baudm/parseq', model_name, pretrained=True, trust_repo=True)
    model = model.eval().to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载成功！")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   设备: {device}")
    print(f"   参数量: {total_params:,}")
    
    return model


def recognize_text(model, image_path, device='cpu'):
    """使用模型识别图片中的文本"""
    from torchvision import transforms
    
    # 加载并预处理图片
    img = Image.open(image_path).convert('RGB')
    
    # PARSeq 需要 128x32 的输入
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 进行推理
    with torch.no_grad():
        output = model(img_tensor)
    
    # 解码结果 (这里简化，实际需要使用模型的 tokenizer)
    # 建议使用项目提供的 read.py 获取完整功能
    return "需要使用项目的 read.py 脚本获取完整的文本识别结果"


def main():
    parser = argparse.ArgumentParser(description='PARSeq Torch Hub 示例')
    parser.add_argument('--model', default='parseq_tiny', 
                       help='模型名称 (默认: parseq_tiny)')
    parser.add_argument('--list-models', action='store_true',
                       help='列出所有可用模型')
    parser.add_argument('--image', type=str,
                       help='要识别的图片路径')
    parser.add_argument('--device', default='cpu',
                       help='设备 (cpu/cuda, 默认: cpu)')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    # 加载模型
    model = load_model(args.model, args.device)
    
    if args.image:
        if os.path.exists(args.image):
            print(f"\n🔍 识别图片: {args.image}")
            result = recognize_text(model, args.image, args.device)
            print(f"识别结果: {result}")
        else:
            print(f"❌ 图片文件不存在: {args.image}")
    
    print(f"\n💡 要获得完整的文本识别功能，请使用:")
    print(f"python read.py pretrained={args.model.replace('_', '-')} --images <图片路径> --device {args.device}")


if __name__ == '__main__':
    main()
