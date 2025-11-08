import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import argparse
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
from pathlib import Path

# FLOPs 계산을 위한 라이브러리
try:
    from ptflops import get_model_complexity_info
    FLOPS_LIBRARY = 'ptflops'
except ImportError:
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        FLOPS_LIBRARY = 'fvcore'
    except ImportError:
        print("⚠️ Warning: Neither ptflops nor fvcore installed. FLOPs calculation will be skipped.")
        print("Install with: pip install ptflops OR pip install fvcore")
        FLOPS_LIBRARY = None

from data_loaders.lol import lowlight_loader
from model.IAT_main import IAT
from IQA_pytorch import SSIM
from utils import PSNR

# Import student model from distillation script
import sys
sys.path.append('.')
from model.IAT_student import IAT_Student_BN


class ModelBenchmark:
    """
    딥러닝 모델의 종합적인 성능 평가 클래스
    - Computational efficiency: Parameters, FLOPs, Inference time
    - Image quality: PSNR, SSIM
    """
    def __init__(self, input_size=(3, 400, 600), device='cuda', warmup_iterations=10):
        self.input_size = input_size
        self.device = device
        self.warmup_iterations = warmup_iterations
        
        # Metrics
        self.ssim = SSIM()
        self.psnr_metric = PSNR()
        
    def count_parameters(self, model):
        """
        모델의 파라미터 수 계산
        Returns:
            total: 전체 파라미터 수
            trainable: 학습 가능한 파라미터 수
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'total_M': total / 1e6,
            'trainable_M': trainable / 1e6
        }
    
    def calculate_flops(self, model, input_size=None):
        """
        FLOPs (Floating Point Operations) 계산
        """
        if input_size is None:
            input_size = self.input_size
        
        if FLOPS_LIBRARY == 'ptflops':
            return self._calculate_flops_ptflops(model, input_size)
        elif FLOPS_LIBRARY == 'fvcore':
            return self._calculate_flops_fvcore(model, input_size)
        else:
            return {'flops': 0, 'flops_G': 0, 'macs': 0, 'macs_G': 0}
    
    def _calculate_flops_ptflops(self, model, input_size):
        """
        Try to calculate FLOPs and params using ptflops.
        Handles models that return multiple outputs (tuple).
        Falls back to manual estimation if ptflops fails.
        """
        try:
            from ptflops import get_model_complexity_info
            import torch.nn as nn

            # ✅ Wrapper: IAT 같은 모델은 (mul, add, enhanced_img) 형태로 출력하므로 마지막 출력만 반환
            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    out = self.model(x)
                    if isinstance(out, (list, tuple)):
                        return out[-1]
                    return out

            wrapped_model = ModelWrapper(model)
        

            # ✅ FLOPs 계산 (verbose=False로 깔끔하게)
            with torch.cuda.device(self.device if self.device == "cuda" else "cpu"):
                macs, params = get_model_complexity_info(
                    wrapped_model,
                    input_size,  # (C,H,W)
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )

            flops = 2 * macs  # 1 MAC = 2 FLOPs
            print(f"✅ FLOPs calculated successfully: {flops/1e9:.2f} GFLOPs, Params: {params/1e6:.2f} M")
            return {
                'flops': flops,
                'flops_G': flops / 1e9,
                'macs': flops / 2,
                'macs_G': flops / 2e9,
                'params': params
            }


        except Exception as e:
            print(f"⚠️ FLOPs calculation failed: {e}")
            # fallback to manual estimation
            return self._calculate_flops_manual(model, input_size)

    
    def _calculate_flops_fvcore(self, model, input_size):
        """fvcore 라이브러리 사용"""
        try:
            dummy_input = torch.randn(1, *input_size).to(self.device)
            flops = FlopCountAnalysis(model, dummy_input)
            total_flops = flops.total()
            
            return {
                'flops': total_flops,
                'flops_G': total_flops / 1e9,
                'macs': total_flops / 2,  # Approximate
                'macs_G': total_flops / 2e9
            }
        except Exception as e:
            print(f"⚠️ FLOPs calculation failed: {e}")
            return self._calculate_flops_manual(model, input_size)
    
    def _calculate_flops_manual(self, model, input_size):
        """
        Manual FLOPs estimation for IAT models
        Conv2d FLOPs = 2 * C_in * C_out * K * K * H_out * W_out
        """
        total_flops = 0
        H, W = input_size[1], input_size[2]
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d FLOPs
                batch_size = 1
                in_c = module.in_channels
                out_c = module.out_channels
                k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                
                # Output spatial dimensions
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                h_out = (H + 2 * padding - k_h) // stride + 1
                w_out = (W + 2 * padding - k_w) // stride + 1
                
                # FLOPs = 2 * C_in * C_out * K_h * K_w * H_out * W_out
                flops_per_conv = 2 * in_c * out_c * k_h * k_w * h_out * w_out
                total_flops += flops_per_conv
                
                # Update H, W for next layer (approximate)
                H, W = h_out, w_out
                
            elif isinstance(module, nn.Linear):
                # Linear FLOPs = 2 * in_features * out_features
                flops_per_linear = 2 * module.in_features * module.out_features
                total_flops += flops_per_linear
        
        return {
            'flops': total_flops,
            'flops_G': total_flops / 1e9,
            'macs': total_flops / 2,
            'macs_G': total_flops / 2e9
        }
    
    def measure_inference_time(self, model, input_size=None, num_iterations=100):
        """
        추론 시간 측정 (GPU/CPU)
        Args:
            model: 측정할 모델
            input_size: 입력 크기 (C, H, W)
            num_iterations: 측정 반복 횟수
        Returns:
            dict: 추론 시간 통계
        """
        if input_size is None:
            input_size = self.input_size
        
        model.eval()
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        # GPU warm-up
        print(f"  Warming up for {self.warmup_iterations} iterations...")
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(dummy_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure inference time
        times = []
        print(f"  Measuring inference time for {num_iterations} iterations...")
        
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc="Inference"):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(dummy_input)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms
        
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'fps': 1000.0 / np.mean(times)
        }
    
    def evaluate_image_quality(self, model, data_loader, max_samples=None):
        """
        이미지 품질 메트릭 평가 (PSNR, SSIM)
        """
        model.eval()
        
        psnr_list = []
        ssim_list = []
        
        with torch.no_grad():
            for i, imgs in enumerate(tqdm(data_loader, desc="Evaluating quality")):
                if max_samples and i >= max_samples:
                    break
                
                low_img = imgs[0].to(self.device)
                high_img = imgs[1].to(self.device)
                
                # Forward pass
                if hasattr(model, 'forward'):
                    outputs = model(low_img)
                    # IAT returns (mul, add, enhanced_img)
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        enhanced = outputs[2]
                    else:
                        enhanced = outputs
                else:
                    enhanced = model(low_img)
                
                # Calculate metrics
                psnr_val = self.psnr_metric(enhanced, high_img)
                ssim_val = self.ssim(enhanced, high_img, as_loss=False)
                
                psnr_list.append(psnr_val.item())
                ssim_list.append(ssim_val.item())
        
        return {
            'psnr_mean': np.mean(psnr_list),
            'psnr_std': np.std(psnr_list),
            'ssim_mean': np.mean(ssim_list),
            'ssim_std': np.std(ssim_list)
        }
    
    def benchmark_model(self, model, model_name, data_loader=None, 
                       input_size=None, inference_iterations=100, max_eval_samples=None):
        """
        모델의 종합적인 벤치마킹
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        
        results = {'model_name': model_name}
        
        # 1. Parameters
        print("\n[1/4] Counting parameters...")
        params_info = self.count_parameters(model)
        results.update(params_info)
        print(f"  Total params: {params_info['total_M']:.2f}M")
        print(f"  Trainable params: {params_info['trainable_M']:.2f}M")
        
        # 2. FLOPs
        print("\n[2/4] Calculating FLOPs...")
        flops_info = self.calculate_flops(model, input_size)
        results.update(flops_info)
        if flops_info['flops_G'] > 0:
            print(f"  FLOPs: {flops_info['flops_G']:.2f}G")
            print(f"  MACs: {flops_info['macs_G']:.2f}G")
        else:
            print(f"  FLOPs: Not available")
        
        # 3. Inference time
        print("\n[3/4] Measuring inference time...")
        time_info = self.measure_inference_time(
            model, input_size, num_iterations=inference_iterations
        )
        results.update(time_info)
        print(f"  Mean: {time_info['mean_ms']:.2f} ± {time_info['std_ms']:.2f} ms")
        print(f"  FPS: {time_info['fps']:.2f}")
        
        # 4. Image quality (optional)
        if data_loader is not None:
            print("\n[4/4] Evaluating image quality...")
            quality_info = self.evaluate_image_quality(
                model, data_loader, max_samples=max_eval_samples
            )
            results.update(quality_info)
            print(f"  PSNR: {quality_info['psnr_mean']:.2f} ± {quality_info['psnr_std']:.2f} dB")
            print(f"  SSIM: {quality_info['ssim_mean']:.4f} ± {quality_info['ssim_std']:.4f}")
        else:
            print("\n[4/4] Skipping image quality evaluation (no data loader provided)")
        
        return results
    
    def compare_models(self, results_list, save_dir='./benchmark_results'):
        """
        여러 모델의 결과를 비교하고 시각화
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Print comparison table
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        # Prepare table data
        headers = ['Metric', 'Unit']
        headers.extend([r['model_name'] for r in results_list])
        
        rows = []
        
        # Parameters
        rows.append(['Parameters', 'M'] + [f"{r['total_M']:.2f}" for r in results_list])
        
        # FLOPs
        if all('flops_G' in r and r['flops_G'] > 0 for r in results_list):
            rows.append(['FLOPs', 'G'] + [f"{r['flops_G']:.2f}" for r in results_list])
            rows.append(['MACs', 'G'] + [f"{r['macs_G']:.2f}" for r in results_list])
        
        # Inference time
        rows.append(['Inference (mean)', 'ms'] + [f"{r['mean_ms']:.2f}" for r in results_list])
        rows.append(['Inference (std)', 'ms'] + [f"{r['std_ms']:.2f}" for r in results_list])
        rows.append(['FPS', 'fps'] + [f"{r['fps']:.2f}" for r in results_list])
        
        # Image quality
        if all('psnr_mean' in r for r in results_list):
            rows.append(['PSNR', 'dB'] + [f"{r['psnr_mean']:.2f}" for r in results_list])
            rows.append(['SSIM', '-'] + [f"{r['ssim_mean']:.4f}" for r in results_list])
        
        # Efficiency metrics
        if all('psnr_mean' in r and 'total_M' in r for r in results_list):
            rows.append(['PSNR/Param', 'dB/M'] + 
                       [f"{r['psnr_mean']/r['total_M']:.2f}" for r in results_list])
        
        if all('fps' in r and 'total_M' in r for r in results_list):
            rows.append(['FPS/Param', 'fps/M'] + 
                       [f"{r['fps']/r['total_M']:.2f}" for r in results_list])
        
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # Calculate compression/speedup ratios
        if len(results_list) == 2:
            teacher_idx = 0 if 'teacher' in results_list[0]['model_name'].lower() else 1
            student_idx = 1 - teacher_idx
            
            teacher = results_list[teacher_idx]
            student = results_list[student_idx]
            
            print(f"\n{'='*80}")
            print("COMPRESSION & SPEEDUP ANALYSIS")
            print(f"{'='*80}\n")
            
            compression_ratio = teacher['total_M'] / student['total_M']
            speedup = teacher['mean_ms'] / student['mean_ms']
            
            print(f"📊 Parameter Compression: {compression_ratio:.2f}x "
                  f"({teacher['total_M']:.2f}M → {student['total_M']:.2f}M)")
            
            if 'flops_G' in teacher and teacher['flops_G'] > 0:
                flops_reduction = teacher['flops_G'] / student['flops_G']
                print(f"⚡ FLOPs Reduction: {flops_reduction:.2f}x "
                      f"({teacher['flops_G']:.2f}G → {student['flops_G']:.2f}G)")
            
            print(f"🚀 Inference Speedup: {speedup:.2f}x "
                  f"({teacher['mean_ms']:.2f}ms → {student['mean_ms']:.2f}ms)")
            print(f"📈 FPS Improvement: {student['fps']:.2f} / {teacher['fps']:.2f} "
                  f"= {student['fps']/teacher['fps']:.2f}x")
            
            if 'psnr_mean' in teacher:
                psnr_diff = student['psnr_mean'] - teacher['psnr_mean']
                ssim_diff = student['ssim_mean'] - teacher['ssim_mean']
                
                print(f"\n📸 Quality Trade-off:")
                print(f"  PSNR: {student['psnr_mean']:.2f} vs {teacher['psnr_mean']:.2f} "
                      f"({psnr_diff:+.2f} dB)")
                print(f"  SSIM: {student['ssim_mean']:.4f} vs {teacher['ssim_mean']:.4f} "
                      f"({ssim_diff:+.4f})")
        
        # 2. Save results to file
        result_file = os.path.join(save_dir, 'benchmark_results.txt')
        with open(result_file, 'w') as f:
            f.write(tabulate(rows, headers=headers, tablefmt='grid'))
            f.write('\n\n')
            
            if len(results_list) == 2:
                f.write(f"Compression Ratio: {compression_ratio:.2f}x\n")
                f.write(f"Speedup: {speedup:.2f}x\n")
                if 'psnr_mean' in teacher:
                    f.write(f"PSNR Difference: {psnr_diff:+.2f} dB\n")
                    f.write(f"SSIM Difference: {ssim_diff:+.4f}\n")
        
        print(f"\n✅ Results saved to {result_file}")
        
        # 3. Visualization
        self._plot_comparison(results_list, save_dir)
        
        return results_list
    
    # def _plot_comparison(self, results_list, save_dir):
    #     """
    #     비교 결과를 시각화
    #     """
    #     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    #     fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
    #     model_names = [r['model_name'] for r in results_list]
        
    #     # 1. Parameters & FLOPs
    #     ax = axes[0, 0]
    #     params = [r['total_M'] for r in results_list]
    #     ax.bar(model_names, params, color=['#3498db', '#e74c3c'])
    #     ax.set_ylabel('Parameters (M)', fontsize=12)
    #     ax.set_title('Model Size', fontsize=13, fontweight='bold')
    #     ax.grid(axis='y', alpha=0.3)
        
    #     for i, v in enumerate(params):
    #         ax.text(i, v + max(params)*0.02, f'{v:.2f}M', 
    #                ha='center', va='bottom', fontweight='bold')
        
    #     # 2. Inference Time
    #     ax = axes[0, 1]
    #     inf_times = [r['mean_ms'] for r in results_list]
    #     ax.bar(model_names, inf_times, color=['#3498db', '#e74c3c'])
    #     ax.set_ylabel('Inference Time (ms)', fontsize=12)
    #     ax.set_title('Inference Speed', fontsize=13, fontweight='bold')
    #     ax.grid(axis='y', alpha=0.3)
        
    #     for i, v in enumerate(inf_times):
    #         ax.text(i, v + max(inf_times)*0.02, f'{v:.2f}ms', 
    #                ha='center', va='bottom', fontweight='bold')
        
    #     # 3. FPS
    #     ax = axes[1, 0]
    #     fps_values = [r['fps'] for r in results_list]
    #     ax.bar(model_names, fps_values, color=['#3498db', '#e74c3c'])
    #     ax.set_ylabel('FPS', fontsize=12)
    #     ax.set_title('Frames Per Second', fontsize=13, fontweight='bold')
    #     ax.grid(axis='y', alpha=0.3)
        
    #     for i, v in enumerate(fps_values):
    #         ax.text(i, v + max(fps_values)*0.02, f'{v:.1f}', 
    #                ha='center', va='bottom', fontweight='bold')
        
    #     # 4. Image Quality (if available)
    #     ax = axes[1, 1]
    #     if all('psnr_mean' in r for r in results_list):
    #         x = np.arange(len(model_names))
    #         width = 0.35
            
    #         psnr_values = [r['psnr_mean'] for r in results_list]
    #         ssim_values = [r['ssim_mean'] * 30 for r in results_list]  # Scale SSIM for visualization
            
    #         bars1 = ax.bar(x - width/2, psnr_values, width, label='PSNR (dB)', color='#2ecc71')
    #         bars2 = ax.bar(x + width/2, ssim_values, width, label='SSIM (×30)', color='#f39c12')
            
    #         ax.set_ylabel('Value', fontsize=12)
    #         ax.set_title('Image Quality Metrics', fontsize=13, fontweight='bold')
    #         ax.set_xticks(x)
    #         ax.set_xticklabels(model_names)
    #         ax.legend()
    #         ax.grid(axis='y', alpha=0.3)
            
    #         for bars in [bars1, bars2]:
    #             for bar in bars:
    #                 height = bar.get_height()
    #                 ax.text(bar.get_x() + bar.get_width()/2., height,
    #                        f'{height:.1f}',
    #                        ha='center', va='bottom', fontsize=9)
    #     else:
    #         ax.text(0.5, 0.5, 'No quality metrics available', 
    #                ha='center', va='center', transform=ax.transAxes,
    #                fontsize=12, color='gray')
    #         ax.axis('off')
        
    #     plt.tight_layout()
        
    #     plot_file = os.path.join(save_dir, 'comparison_plot.png')
    #     plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    #     print(f"✅ Comparison plot saved to {plot_file}")
    #     plt.close()
    def _plot_comparison(self, results_list, save_dir):
        """
        비교 결과를 시각화하고 각각의 그래프를 개별 파일로 저장
        """
        model_names = [r['model_name'] for r in results_list]
        colors = ['#3498db', '#e74c3c'] # 색상 팔레트 정의

        # 1. Parameters & FLOPs
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        params = [r['total_M'] for r in results_list]
        flops = [r.get('flops_G', 0) for r in results_list] # FLOPs가 없을 경우 0으로 처리

        # 첫 번째 y축: Parameters
        ax1.bar(model_names, params, width=0.35, label='Parameters (M)', color=colors[0])
        ax1.set_ylabel('Parameters (M)', fontsize=12, color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])
        ax1.set_title('Model Size', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for i, v in enumerate(params):
            ax1.text(i, v + max(params)*0.02, f'{v:.2f}M',
                       ha='center', va='bottom', fontweight='bold', color=colors[0])

        # 두 번째 y축: FLOPs (FLOPs 데이터가 있을 경우에만)
        if any(f > 0 for f in flops):
            ax1_twin = ax1.twinx()
            ax1_twin.bar([p + 0.35 for p in range(len(model_names))], flops, width=0.35, label='FLOPs (G)', color=colors[1])
            ax1_twin.set_ylabel('FLOPs (G)', fontsize=12, color=colors[1])
            ax1_twin.tick_params(axis='y', labelcolor=colors[1])
            for i, v in enumerate(flops):
                if v > 0:
                    ax1_twin.text(i + 0.35, v + max(flops)*0.02, f'{v:.2f}G',
                                  ha='center', va='bottom', fontweight='bold', color=colors[1])
            # 범례 합치기
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1_twin.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')


        plt.tight_layout()
        plot_file1 = os.path.join(save_dir, 'model_size_comparison.png')
        fig1.savefig(plot_file1, dpi=300, bbox_inches='tight')
        print(f"✅ Model size comparison plot saved to {plot_file1}")
        plt.close(fig1)

        # 2. Inference Time
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        inf_times = [r['mean_ms'] for r in results_list]
        ax2.bar(model_names, inf_times, color=colors)
        ax2.set_ylabel('Inference Time (ms)', fontsize=12)
        ax2.set_title('Inference Speed (Mean)', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for i, v in enumerate(inf_times):
            ax2.text(i, v + max(inf_times)*0.02, f'{v:.2f}ms',
                       ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plot_file2 = os.path.join(save_dir, 'inference_time_comparison.png')
        fig2.savefig(plot_file2, dpi=300, bbox_inches='tight')
        print(f"✅ Inference time comparison plot saved to {plot_file2}")
        plt.close(fig2)

        # 3. FPS
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        fps_values = [r['fps'] for r in results_list]
        ax3.bar(model_names, fps_values, color=colors)
        ax3.set_ylabel('FPS', fontsize=12)
        ax3.set_title('Frames Per Second', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        for i, v in enumerate(fps_values):
            ax3.text(i, v + max(fps_values)*0.02, f'{v:.1f}',
                       ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plot_file3 = os.path.join(save_dir, 'fps_comparison.png')
        fig3.savefig(plot_file3, dpi=300, bbox_inches='tight')
        print(f"✅ FPS comparison plot saved to {plot_file3}")
        plt.close(fig3)

        # 4. Image Quality (if available)
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        if all('psnr_mean' in r for r in results_list):
            x = np.arange(len(model_names))
            width = 0.35

            psnr_values = [r['psnr_mean'] for r in results_list]
            ssim_values_scaled = [r['ssim_mean'] * 30 for r in results_list]
            
            bars1 = ax4.bar(x - width/2, psnr_values, width, label='PSNR (dB)', color='#2ecc71')
            bars2 = ax4.bar(x + width/2, ssim_values_scaled, width, label='SSIM(x30)', color='#f39c12')

            ax4.set_ylabel('Value', fontsize=12)
            ax4.set_title('Image Quality Metrics', fontsize=13, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(model_names)
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'No quality metrics available',
                   ha='center', va='center', transform=ax4.transAxes,
                   fontsize=12, color='gray')
            ax4.axis('off')

        plt.tight_layout()
        plot_file4 = os.path.join(save_dir, 'image_quality_comparison.png')
        fig4.savefig(plot_file4, dpi=300, bbox_inches='tight')
        print(f"✅ Image quality comparison plot saved to {plot_file4}")
        plt.close(fig4)

def main():
    parser = argparse.ArgumentParser(description='Benchmark IAT Teacher and Student models')
    
    # Model paths
    parser.add_argument('--teacher_path', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--student_path', type=str, required=True,
                       help='Path to student model checkpoint')
    
    # Data
    parser.add_argument('--test_data_path', type=str, 
                       default="/content/drive/MyDrive/LOL-v2/Real_captured/Test/Low/",
                       help='Path to test dataset')
    parser.add_argument('--evaluate_quality', action='store_true',
                       help='Evaluate image quality metrics (PSNR, SSIM)')
    parser.add_argument('--max_eval_samples', type=int, default=100,
                       help='Maximum number of samples for quality evaluation')
    
    # Benchmark settings
    parser.add_argument('--input_size', type=int, nargs=3, default=[3, 400, 600],
                       help='Input size (C, H, W)')
    parser.add_argument('--inference_iterations', type=int, default=100,
                       help='Number of iterations for inference time measurement')
    parser.add_argument('--warmup_iterations', type=int, default=10,
                       help='Number of warmup iterations')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./benchmark_results_01',
                       help='Directory to save results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run benchmark')
    parser.add_argument('--model_type', type=str, default='lol',
                       help='Model type (lol/lol_v2)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("IAT Teacher-Student Model Benchmark")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Input size: {args.input_size}")
    print(f"Inference iterations: {args.inference_iterations}")
    print(f"Evaluate quality: {args.evaluate_quality}")
    
    # Initialize benchmark
    benchmark = ModelBenchmark(
        input_size=tuple(args.input_size),
        device=args.device,
        warmup_iterations=args.warmup_iterations
    )
    
    # Load data loader (if quality evaluation is needed)
    test_loader = None
    if args.evaluate_quality:
        print(f"\nLoading test dataset from {args.test_data_path}")
        test_dataset = lowlight_loader(
            images_path=args.test_data_path,
            mode='test',
            normalize=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"Test dataset size: {len(test_dataset)}")
    
    # ==================== Teacher Model ====================
    print(f"\nLoading Teacher model from {args.teacher_path}")
    teacher = IAT(type=args.model_type, with_global=True).to(args.device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=args.device))
    teacher.eval()
    
    teacher_results = benchmark.benchmark_model(
        model=teacher,
        model_name='Teacher (IAT Full)',
        data_loader=test_loader,
        input_size=tuple(args.input_size),
        inference_iterations=args.inference_iterations,
        max_eval_samples=args.max_eval_samples if args.evaluate_quality else None
    )
    
    # ==================== Student Model ====================
    print(f"\nLoading Student model from {args.student_path}")
    # student = IAT_Student_BN().to(args.device)
    # student.load_state_dict(torch.load(args.student_path, map_location=args.device, weights_only=False))
    student = IAT_Student_BN().to(args.device)
    checkpoint = torch.load(args.student_path, map_location=args.device, weights_only=False)
    student.load_state_dict(checkpoint['state_dict'])
    
    student.eval()
    
    student_results = benchmark.benchmark_model(
        model=student,
        model_name='Student (Distilled)',
        data_loader=test_loader,
        input_size=tuple(args.input_size),
        inference_iterations=args.inference_iterations,
        max_eval_samples=args.max_eval_samples if args.evaluate_quality else None
    )
    
    # ==================== Comparison ====================
    results = benchmark.compare_models(
        [teacher_results, student_results],
        save_dir=args.save_dir
    )
    
    print(f"\n{'='*80}")
    print("✅ Benchmark completed!")
    print(f"Results saved to {args.save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()