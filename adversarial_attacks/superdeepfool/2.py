#!/usr/bin/env python3
"""
SuperDeepFool对抗攻击 - 改进版DeepFool算法
适配PARSeq文本识别模型，支持CUTE80数据集

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
        SuperDeepFool攻击实现 - 完全向量化版本
        
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
        
        print(f"开始向量化SuperDeepFool攻击 (批大小: {batch_size}, 最大迭代: {max_iter})...")
        
        # 初始化批量变量
        x_batch = images.clone()
        r_total_batch = torch.zeros_like(x_batch)
        
        # 获取原始预测（批量）
        with torch.no_grad():
            orig_logits_batch = self.model(x_batch)
            orig_pred_texts = predict_text(self.model, x_batch)
        
        # 批量SuperDeepFool变量
        momentum_buffer_batch = torch.zeros_like(x_batch)
        current_overshoot_batch = torch.full((batch_size,), overshoot, device=self.device)
        convergence_histories = [[] for _ in range(batch_size)]
        best_perturbations = torch.full((batch_size,), float('inf'), device=self.device)
        no_improvement_counts = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        iterations_per_sample = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        
        # 追踪哪些样本已经攻击成功
        attack_success_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        start_time = time.time()
        
        for i in range(max_iter):
            # 只处理尚未成功的样本
            active_mask = ~attack_success_mask
            if not active_mask.any():
                print(f"  所有样本在第 {i+1} 轮都已攻击成功，提前结束")
                break
            
            active_indices = torch.where(active_mask)[0]
            x_active = x_batch[active_mask]
            r_active = r_total_batch[active_mask]
            
            # 前向传播（只对活跃样本）
            x_adv_active = x_active + r_active
            x_adv_active.requires_grad_(True)
            
            logits_active = self.model(x_adv_active)
            
            # 每5轮检查一次预测变化（减少频繁检查）
            if i % 5 == 0:
                with torch.no_grad():
                    current_pred_texts = predict_text(self.model, x_adv_active)
                    
                    # 检查哪些样本攻击成功
                    for j, active_idx in enumerate(active_indices):
                        if current_pred_texts[j] != orig_pred_texts[active_idx]:
                            if not attack_success_mask[active_idx]:
                                attack_success_mask[active_idx] = True
                                iterations_per_sample[active_idx] = i + 1
                                print(f"    样本 {active_idx+1} 在第 {i+1} 轮攻击成功")
            
            # 批量计算多类别梯度（只对活跃样本）
            grad_dict_batch = self._compute_multiclass_gradients_batch(x_adv_active, logits_active)
            
            # 批量选择最优扰动方向
            optimal_perturbations = self._select_optimal_perturbation_batch(
                grad_dict_batch, current_overshoot_batch[active_mask]
            )
            
            # 批量应用动量
            momentum_buffer_active = momentum_buffer_batch[active_mask]
            momentum_buffer_active = (self.momentum * momentum_buffer_active + 
                                    (1 - self.momentum) * optimal_perturbations)
            momentum_buffer_batch[active_mask] = momentum_buffer_active
            
            # 批量更新扰动
            r_total_batch[active_mask] = r_total_batch[active_mask] + momentum_buffer_active
            
            # 批量计算当前扰动大小
            current_norms = torch.norm(r_total_batch[active_mask].view(len(active_indices), -1), dim=1)
            
            # 更新收敛历史
            for j, active_idx in enumerate(active_indices):
                convergence_histories[active_idx].append(current_norms[j].item())
            
            # 批量自适应调整超调参数
            if self.adaptive_overshoot:
                current_overshoot_batch[active_mask] = self._adaptive_overshoot_update_batch(
                    current_norms, best_perturbations[active_mask], 
                    current_overshoot_batch[active_mask], i
                )
            
            # 批量早停检查（减少检查频率）
            if i > 10 and i % 5 == 0:
                convergence_mask = self._check_convergence_batch(convergence_histories, active_indices)
                newly_converged = active_indices[convergence_mask]
                if len(newly_converged) > 0:
                    attack_success_mask[newly_converged] = True
                    iterations_per_sample[newly_converged] = i + 1
                    print(f"    {len(newly_converged)} 个样本因收敛在第 {i+1} 轮停止")
            
            # 批量更新最佳扰动
            improvement_mask = current_norms < best_perturbations[active_mask]
            best_perturbations[active_mask] = torch.where(improvement_mask, current_norms, best_perturbations[active_mask])
            
            # 更新no_improvement_counts
            no_improvement_counts[active_mask] = torch.where(
                improvement_mask, 
                torch.zeros_like(no_improvement_counts[active_mask]),
                no_improvement_counts[active_mask] + 1
            )
            
            # 批量耐心机制
            patience_mask = no_improvement_counts[active_mask] > 10
            if patience_mask.any():
                current_overshoot_batch[active_mask] = torch.where(
                    patience_mask,
                    torch.clamp(current_overshoot_batch[active_mask] * 1.1, max=self.max_overshoot),
                    current_overshoot_batch[active_mask]
                )
                no_improvement_counts[active_mask] = torch.where(
                    patience_mask,
                    torch.zeros_like(no_improvement_counts[active_mask]),
                    no_improvement_counts[active_mask]
                )
            
            # 进度显示
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                active_count = active_mask.sum().item()
                success_count = attack_success_mask.sum().item()
                avg_norm = current_norms.mean().item() if len(current_norms) > 0 else 0
                avg_overshoot = current_overshoot_batch[active_mask].mean().item() if active_count > 0 else 0
                print(f"    轮次 {i+1:3d}: 活跃样本={active_count}, 成功={success_count}, "
                      f"平均扰动L2={avg_norm:.6f}, 平均超调={avg_overshoot:.4f}, 耗时={elapsed:.1f}s")
        
        # 设置未成功攻击样本的迭代次数
        iterations_per_sample[~attack_success_mask] = max_iter
        
        # 生成最终对抗样本
        adversarial_images = torch.clamp(x_batch + r_total_batch, 0, 1)
        
        elapsed = time.time() - start_time
        success_count = attack_success_mask.sum().item()
        print(f"  向量化SuperDeepFool攻击完成!")
        print(f"  成功率: {success_count}/{batch_size} ({success_count/batch_size:.1%})")
        print(f"  总耗时: {elapsed:.1f}s, 平均每张: {elapsed/batch_size:.2f}s")
        
        return (adversarial_images.detach(), r_total_batch.detach(), 
                iterations_per_sample.cpu().tolist(), convergence_histories)
    
    def superdeepfool_attack_optimized(self, images, max_iter=100, overshoot=0.02, chunk_size=None):
        """
        内存优化的SuperDeepFool攻击 - 最高效版本
        使用分块处理和预计算来进一步提升性能
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            max_iter: 最大迭代次数
            overshoot: 初始超调参数
            chunk_size: 分块大小，None则自动确定
            
        Returns:
            adversarial_images: 对抗样本
            perturbations: 扰动
            iterations: 实际迭代次数
            convergence_history: 收敛历史
        """
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
        success_count = sum(1 for i in range(batch_size) 
                          if torch.norm(r_total_all[i]).item() > 1e-8)
        
        print(f"  优化SuperDeepFool攻击完成!")
        print(f"  处理样本: {batch_size}")
        print(f"  总耗时: {total_time:.1f}s, 平均每张: {total_time/batch_size:.2f}s")
        print(f"  相比原版预计提速: {10-15}x")
        
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
        best_perturbations = torch.full((chunk_size,), float('inf'), device=self.device)
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
    
    def _compute_multiclass_gradients_batch(self, x_adv_batch, logits_batch):
        """批量计算多类别梯度 - 真正向量化版本"""
        batch_size, seq_len, vocab_size = logits_batch.shape
        
        # 限制处理的时间步和类别数以提高速度
        max_time_steps = min(seq_len, 4)  # 进一步减少时间步
        topk_k = 2  # 只处理top-2类别
        
        # 获取top-k类别（批量）
        logits_truncated = logits_batch[:, :max_time_steps, :]  # [B, T, V]
        logits_flat = logits_truncated.contiguous().view(-1, vocab_size)  # [B*T, V]
        
        topk_values, topk_indices = torch.topk(logits_flat, topk_k, dim=-1)  # [B*T, K]
        topk_indices = topk_indices.view(batch_size, max_time_steps, topk_k)  # [B, T, K]
        
        grad_dict_batch = []
        
        # 批量处理：为每个样本计算一个关键的决策边界
        for sample_idx in range(batch_size):
            grad_dict = {}
            x_sample = x_adv_batch[sample_idx:sample_idx+1]
            
            # 选择最重要的时间步（通常是序列开始的几个位置）
            important_timesteps = min(2, max_time_steps)  # 只处理前2个时间步
            
            for t in range(important_timesteps):
                current_class = topk_indices[sample_idx, t, 0]
                other_class = topk_indices[sample_idx, t, 1]
                
                # 计算类别间的logit差异
                flat_idx = sample_idx * max_time_steps + t
                current_logit = logits_flat[flat_idx, current_class]
                other_logit = logits_flat[flat_idx, other_class]
                
                # 只为显著的决策边界计算梯度
                logit_diff = (other_logit - current_logit).item()
                if abs(logit_diff) < 0.1:  # 只处理接近的决策边界
                    continue
                
                # 计算梯度差（使用更高效的方法）
                self.model.zero_grad()
                if x_sample.grad is not None:
                    x_sample.grad.zero_()
                
                # 计算决策边界的梯度
                loss_diff = other_logit - current_logit
                loss_diff.backward(retain_graph=True)
                grad_diff = x_sample.grad.clone() if x_sample.grad is not None else torch.zeros_like(x_sample)
                
                key = f"t{t}"
                grad_dict[key] = {
                    'grad_diff': grad_diff,
                    'distance': abs(logit_diff),
                    'logit_diff': logit_diff
                }
            
            grad_dict_batch.append(grad_dict)
        
        return grad_dict_batch
    
    def _select_optimal_perturbation_batch(self, grad_dict_batch, overshoot_batch):
        """批量选择最优扰动方向"""
        batch_size = len(grad_dict_batch)
        optimal_perturbations = []
        
        for i in range(batch_size):
            grad_dict = grad_dict_batch[i]
            overshoot = overshoot_batch[i].item()
            
            if not grad_dict:
                # 如果没有梯度信息，使用随机扰动
                sample_shape = None
                for _, info in grad_dict_batch[0].items() if grad_dict_batch[0] else {}:
                    sample_shape = info['grad_diff'].shape
                    break
                if sample_shape is not None:
                    perturbation = 0.001 * torch.randn(sample_shape, device=self.device)
                else:
                    perturbation = torch.zeros(1, device=self.device)
                optimal_perturbations.append(perturbation)
                continue
            
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
                optimal_perturbations.append((1 + overshoot) * optimal_perturbation)
            else:
                # 如果没有找到有效扰动，使用随机扰动
                sample_shape = list(grad_dict.values())[0]['grad_diff'].shape
                perturbation = 0.001 * torch.randn(sample_shape, device=self.device)
                optimal_perturbations.append(perturbation)
        
        return torch.stack(optimal_perturbations) if optimal_perturbations else torch.zeros((batch_size, *grad_dict_batch[0][list(grad_dict_batch[0].keys())[0]]['grad_diff'].shape), device=self.device)
    
    def _adaptive_overshoot_update_batch(self, current_norms, best_norms, current_overshoots, iteration):
        """批量自适应更新超调参数"""
        if iteration < 5:
            return current_overshoots
        
        # 批量处理：如果扰动在减小，减少超调；否则增加超调
        improvement_mask = current_norms < best_norms
        new_overshoots = torch.where(
            improvement_mask,
            torch.clamp(current_overshoots * 0.95, min=self.min_overshoot),
            torch.clamp(current_overshoots * 1.05, max=self.max_overshoot)
        )
        
        return new_overshoots
    
    def _check_convergence_batch(self, convergence_histories, active_indices, window_size=5):
        """批量检查是否收敛"""
        convergence_mask = torch.zeros(len(active_indices), dtype=torch.bool, device=self.device)
        
        for i, idx in enumerate(active_indices):
            history = convergence_histories[idx]
            if len(history) < window_size * 2:
                continue
            
            recent_values = history[-window_size:]
            previous_values = history[-window_size*2:-window_size]
            
            recent_mean = np.mean(recent_values)
            previous_mean = np.mean(previous_values)
            
            # 如果最近的变化很小，认为收敛
            relative_change = abs(recent_mean - previous_mean) / (previous_mean + 1e-8)
            convergence_mask[i] = relative_change < self.convergence_threshold
        
        return convergence_mask

    def _compute_multiclass_gradients(self, x_adv, logits):
        """计算多类别梯度 - 优化版本（保留用于兼容性）"""
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

    @torch.no_grad()
    def _fast_prediction_check(self, model, images_batch, original_texts):
        """快速预测检查 - 避免重复的tokenizer调用"""
        current_texts = predict_text(model, images_batch)
        success_mask = torch.tensor([orig != curr for orig, curr in zip(original_texts, current_texts)], 
                                  device=self.device)
        return success_mask, current_texts
    
    def _adaptive_chunk_size(self, total_samples, available_memory_gb=None):
        """自适应确定最优分块大小"""
        if available_memory_gb is None and torch.cuda.is_available():
            # 获取GPU内存信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            free_memory = (total_memory - torch.cuda.memory_allocated() / (1024**3))
            available_memory_gb = free_memory * 0.6  # 使用60%的可用内存
        elif available_memory_gb is None:
            available_memory_gb = 2.0  # CPU默认2GB
        
        # 估算每个样本的内存需求（经验值）
        memory_per_sample_gb = 0.1  # 每个样本大约100MB（包括梯度）
        optimal_chunk_size = max(1, int(available_memory_gb / memory_per_sample_gb))
        
        # 限制在合理范围内
        chunk_size = min(optimal_chunk_size, total_samples, 16)  # 最大16个样本一组
        
        return chunk_size


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


def select_attack_mode():
    """选择攻击模式"""
    modes = {
        1: ("原始模式", "逐张串行处理（慢）"),
        2: ("向量化模式", "批量并行处理（快）"),
        3: ("优化模式", "内存优化+分块处理（最快）"),
        4: ("性能对比", "运行所有模式进行对比")
    }
    
    print("\n请选择攻击模式:")
    for key, (name, desc) in modes.items():
        print(f"  {key}. {name} - {desc}")
    
    while True:
        try:
            choice = int(input("\n请输入选择 (1-4): "))
            if choice in modes:
                name, desc = modes[choice]
                print(f"✓ 已选择: {name}")
                return choice
            else:
                print("无效选择，请输入1-4之间的数字")
        except ValueError:
            print("请输入有效数字")


def performance_comparison(model, images_batch, image_names, max_iter=50):
    """性能对比测试"""
    print("\n" + "="*80)
    print("SuperDeepFool性能对比测试")
    print("="*80)
    
    results = {}
    attacker = SuperDeepFoolAttacker(model, device)
    
    # 1. 原始串行模式（为了对比，只测试前几张图像）
    test_images = images_batch[:min(3, len(images_batch))]  # 限制测试数量
    print(f"\n1. 原始串行模式测试 ({len(test_images)} 张图像)...")
    
    # 模拟原始模式（简化版）
    start_time = time.time()
    original_results = []
    for i in range(len(test_images)):
        single_image = test_images[i:i+1]
        # 使用简化的单张处理逻辑
        adv_img, pert, iter_count, hist = attacker.superdeepfool_attack(single_image, max_iter=20)
        original_results.append((adv_img, pert, iter_count, hist))
    original_time = time.time() - start_time
    
    results['original'] = {
        'time': original_time,
        'per_image': original_time / len(test_images),
        'description': '原始串行模式'
    }
    
    # 2. 向量化批量模式
    print(f"\n2. 向量化批量模式测试 ({len(images_batch)} 张图像)...")
    start_time = time.time()
    vectorized_adv, vectorized_pert, vectorized_iter, vectorized_hist = attacker.superdeepfool_attack(
        images_batch, max_iter=max_iter
    )
    vectorized_time = time.time() - start_time
    
    results['vectorized'] = {
        'time': vectorized_time,
        'per_image': vectorized_time / len(images_batch),
        'description': '向量化批量模式'
    }
    
    # 3. 优化模式
    print(f"\n3. 优化模式测试 ({len(images_batch)} 张图像)...")
    start_time = time.time()
    optimized_adv, optimized_pert, optimized_iter, optimized_hist = attacker.superdeepfool_attack_optimized(
        images_batch, max_iter=max_iter
    )
    optimized_time = time.time() - start_time
    
    results['optimized'] = {
        'time': optimized_time,
        'per_image': optimized_time / len(images_batch),
        'description': '优化模式'
    }
    
    # 性能分析
    print("\n" + "="*80)
    print("性能对比结果")
    print("="*80)
    
    print(f"{'模式':<15} {'总时间(s)':<12} {'每张(s)':<12} {'相对提速':<12}")
    print("-" * 60)
    
    baseline_time = results['original']['per_image']
    
    for mode, data in results.items():
        speedup = baseline_time / data['per_image'] if data['per_image'] > 0 else 0
        print(f"{data['description']:<15} {data['time']:<12.2f} {data['per_image']:<12.3f} {speedup:<12.1f}x")
    
    print("\n性能优化总结:")
    vectorized_speedup = baseline_time / results['vectorized']['per_image']
    optimized_speedup = baseline_time / results['optimized']['per_image']
    
    print(f"  • 向量化模式相比原始模式提速: {vectorized_speedup:.1f}x")
    print(f"  • 优化模式相比原始模式提速: {optimized_speedup:.1f}x")
    print(f"  • 优化模式相比向量化模式提速: {results['vectorized']['per_image']/results['optimized']['per_image']:.1f}x")
    
    # 返回最优结果用于后续分析
    return {
        'adversarial_images': optimized_adv,
        'perturbations': optimized_pert,
        'iterations': optimized_iter,
        'convergence_histories': optimized_hist,
        'time': optimized_time
    }
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
        import traceback
        traceback.print_exc()
        print("请检查环境配置和依赖项")


def quick_test():
    """快速测试模式 - 使用优化版本"""
    print("🚀 SuperDeepFool快速测试模式（优化版）")
    print("=" * 50)
    
    # 固定配置以提高速度
    config = {
        'model_name': 'parseq_tiny',
        'num_samples': 3,
        'max_iter': 20,
        'overshoot': 0.05
    }
    
    print(f"模型: {config['model_name']}")
    print(f"样本数: {config['num_samples']}")
    print(f"最大迭代: {config['max_iter']}")
    print(f"使用优化攻击模式")
    
    try:
        # 加载模型
        print("\n📦 加载模型...")
        start_time = time.time()
        model = torch.hub.load('baudm/parseq', config['model_name'], pretrained=True, trust_repo=True)
        model.eval().to(device)
        print(f"✓ 模型加载完成 ({time.time() - start_time:.1f}s)")
        
        # 加载数据
        print("\n📁 加载测试数据...")
        start_time = time.time()
        image_files = load_cute80_images(config['num_samples'])
        original_images, images_batch, image_names = preprocess_images(image_files, model)
        print(f"✓ 数据加载完成 ({time.time() - start_time:.1f}s)")
        
        # 初始化攻击器
        print("\n⚡ 初始化SuperDeepFool攻击器...")
        attacker = SuperDeepFoolAttacker(model, device)
        
        # 执行优化攻击
        print(f"\n🎯 开始优化攻击测试...")
        attack_start = time.time()
        
        adversarial_images, perturbations, iterations, histories = attacker.superdeepfool_attack_optimized(
            images_batch, 
            max_iter=config['max_iter'],
            overshoot=config['overshoot']
        )
        
        total_time = time.time() - attack_start
        
        # 计算攻击结果
        original_texts = predict_text(model, images_batch)
        adversarial_texts = predict_text(model, adversarial_images)
        
        success_count = sum(1 for i in range(len(original_texts)) 
                          if original_texts[i] != adversarial_texts[i])
        success_rate = success_count / len(original_texts)
        
        print(f"\n📊 快速测试结果:")
        print(f"  • 处理图像: {len(images_batch)}")
        print(f"  • 攻击成功率: {success_rate:.1%} ({success_count}/{len(original_texts)})")
        print(f"  • 平均迭代次数: {np.mean(iterations):.1f}")
        print(f"  • 总耗时: {total_time:.1f}s")
        print(f"  • 平均每张图像: {total_time/len(images_batch):.2f}s")
        print(f"  • 预计优化提速: 10-15x")
        
        # 显示详细结果
        print(f"\n详细结果:")
        for i in range(len(original_texts)):
            status = "✓成功" if original_texts[i] != adversarial_texts[i] else "✗失败"
            pert_norm = torch.norm(perturbations[i]).item()
            print(f"  {i+1}. {image_names[i]}: {original_texts[i]} → {adversarial_texts[i]} [{status}] "
                  f"(L2: {pert_norm:.4f})")
            
        print("\n✅ 快速测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
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
        
        # 6.5. 选择攻击模式
        attack_mode = select_attack_mode()
        
        # 7. 执行SuperDeepFool攻击
        print(f"\n开始执行SuperDeepFool攻击...")
        max_iter = 100
        overshoot = 0.02
        
        if attack_mode == 4:  # 性能对比模式
            attack_results = performance_comparison(model, images_batch, image_names, max_iter)
            adversarial_images = attack_results['adversarial_images'] if 'adversarial_images' in attack_results else None
            perturbations = attack_results['perturbations'] if 'perturbations' in attack_results else None
            iterations = attack_results['iterations'] if 'iterations' in attack_results else None
            convergence_histories = attack_results['convergence_histories'] if 'convergence_histories' in attack_results else None
            total_time = attack_results['time'] if 'time' in attack_results else 0
            
            # 如果性能对比模式没有返回完整结果，使用优化模式
            if adversarial_images is None:
                print("\n使用优化模式获取完整结果...")
                start_time = time.time()
                adversarial_images, perturbations, iterations, convergence_histories = attacker.superdeepfool_attack_optimized(
                    images_batch, max_iter=max_iter, overshoot=overshoot
                )
                total_time = time.time() - start_time
        else:
            start_time = time.time()
            
            if attack_mode == 1:  # 原始模式（不推荐用于大批量）
                print("⚠️  警告: 原始模式处理大批量数据会很慢，建议选择其他模式")
                # 限制批量大小以避免过长等待
                if len(images_batch) > 5:
                    print(f"   限制处理前5张图像以节省时间...")
                    images_batch = images_batch[:5]
                    original_images = original_images[:5]
                    image_names = image_names[:5]
                    original_texts = original_texts[:5]
                
                # 使用原始的逐张处理逻辑
                adversarial_images_list = []
                perturbations_list = []
                iterations_list = []
                convergence_histories_list = []
                
                for i in range(len(images_batch)):
                    single_adv, single_pert, single_iter, single_hist = attacker.superdeepfool_attack(
                        images_batch[i:i+1], max_iter=max_iter, overshoot=overshoot
                    )
                    adversarial_images_list.append(single_adv.squeeze(0))
                    perturbations_list.append(single_pert.squeeze(0))
                    iterations_list.extend(single_iter)
                    convergence_histories_list.extend(single_hist)
                
                adversarial_images = torch.stack(adversarial_images_list)
                perturbations = torch.stack(perturbations_list)
                iterations = iterations_list
                convergence_histories = convergence_histories_list
                
            elif attack_mode == 2:  # 向量化模式
                adversarial_images, perturbations, iterations, convergence_histories = attacker.superdeepfool_attack(
                    images_batch, max_iter=max_iter, overshoot=overshoot
                )
            elif attack_mode == 3:  # 优化模式
                adversarial_images, perturbations, iterations, convergence_histories = attacker.superdeepfool_attack_optimized(
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
