"""
Student vs Teacher: Intermediate Results Comparison

중간 단계별 결과 비교:
1. Input (low-light image)
2. After mul (곱셈 적용 후)
3. After add (덧셈 적용 후, img_local)
4. After color matrix (색상 보정 후)
5. Final output (gamma 적용 후)
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from model.IAT_main import IAT
from model.IAT_student import IAT_Student_BN

def extract_teacher_intermediates(teacher, img_low):
    """Teacher의 모든 중간 결과 추출"""
    with torch.no_grad():
        # Teacher forward pass
        mul_t, add_t, img_high_t = teacher(img_low)
        
        # Step 1: After mul
        img_after_mul_t = img_low * mul_t
        
        # Step 2: After add (img_local)
        img_local_t = img_after_mul_t + add_t
        
        # For global parameters, we need to get them from the model
        # Teacher's global_net returns gamma and color directly
        if hasattr(teacher, 'global_net'):
            gamma_t, color_t = teacher.global_net(img_low)
        else:
            # If global_net is not accessible, extract from forward pass
            # The teacher applies: apply_color(img_local) ** gamma
            # We need to reverse engineer this
            gamma_t = torch.ones(img_low.shape[0], 1, 1, 1).to(img_low.device)
            color_t = torch.eye(3).unsqueeze(0).repeat(img_low.shape[0], 1, 1).to(img_low.device)
        
        # Step 3: After color matrix
        # Apply color matrix to img_local
        b, c, h, w = img_local_t.shape
        img_local_flat = img_local_t.view(b, c, -1)
        img_after_color_t = torch.bmm(color_t, img_local_flat)
        img_after_color_t = img_after_color_t.view(b, c, h, w)
        img_after_color_t = torch.clamp(img_after_color_t, 1e-8, 1.0)
        
        # Step 4: Final (already computed by teacher forward)
        # img_high_t is the final result
    
    return {
        'input': img_low,
        'mul': mul_t,
        'add': add_t,
        'after_mul': img_after_mul_t,
        'after_add': img_local_t,
        'after_color': img_after_color_t,
        'final': img_high_t,
        'gamma': gamma_t if gamma_t.dim() == 2 else gamma_t.view(-1, 1),
        'color': color_t
    }


# ============================================================================
# Visualization
# ============================================================================

def tensor_to_image(tensor):
    """Tensor [1, C, H, W] → numpy [H, W, C]"""
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    # 값의 범위를 [0, 1]로 clipping (일부 중간 결과는 범위를 벗어날 수 있음)
    img = np.clip(img, 0, 1)
    return img


def save_comparison_grid(student_results, teacher_results, save_path):
    """Student vs Teacher 비교 그리드 저장"""
    stages = ['input', 'after_mul', 'after_add', 'after_color', 'final']
    stage_names = [
        'Input\n(Low-Light)',
        'After Mul\n(×)',
        'After Add\n(× + add)',
        'After Color\n(color matrix)',
        'Final\n(gamma)'
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for col, (stage, name) in enumerate(zip(stages, stage_names)):
        # Student (row 0)
        student_img = tensor_to_image(student_results[stage])
        axes[0, col].imshow(student_img)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_title(f'STUDENT\n{name}', fontsize=12, weight='bold')
        else:
            axes[0, col].set_title(name, fontsize=11)
        
        # Teacher (row 1)
        teacher_img = tensor_to_image(teacher_results[stage])
        axes[1, col].imshow(teacher_img)
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_title(f'TEACHER\n{name}', fontsize=12, weight='bold')
        else:
            axes[1, col].set_title(name, fontsize=11)
    
    plt.suptitle('Student vs Teacher: Stage-by-Stage Comparison', 
                 fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparison grid saved: {save_path}")
    plt.close()


def save_individual_images(results, prefix, save_dir):
    """개별 이미지로 저장 (torchvision 사용)"""
    import torchvision
    
    stages = {
        'input': 'input',
        'after_mul': 'after_mul',
        'after_add': 'after_add_local',
        'after_color': 'after_color_matrix',
        'final': 'final'
    }
    
    for key, filename in stages.items():
        save_path = os.path.join(save_dir, f"{prefix}_{filename}.png")
        # torchvision.utils.save_image는 자동으로 [0,1] clipping
        torchvision.utils.save_image(results[key], save_path)
        print(f"  Saved: {prefix}_{filename}.png")


def print_parameter_comparison(student_results, teacher_results):
    """Gamma와 Color Matrix 값 비교"""
    print("\n" + "="*70)
    print("Parameter Comparison")
    print("="*70)
    
    # Gamma
    gamma_s = student_results['gamma'].item()
    gamma_t = teacher_results['gamma'].item()
    print(f"\nGamma:")
    print(f"  Student: {gamma_s:.6f}")
    print(f"  Teacher: {gamma_t:.6f}")
    print(f"  Diff:    {abs(gamma_s - gamma_t):.6f}")
    
    # Color Matrix
    color_s = student_results['color'].squeeze(0).cpu().numpy()
    color_t = teacher_results['color'].squeeze(0).cpu().numpy()
    
    print(f"\nColor Matrix (3×3):")
    print(f"  Student:")
    for row in color_s:
        print(f"    {row}")
    print(f"  Teacher:")
    for row in color_t:
        print(f"    {row}")
    
    diff = np.abs(color_s - color_t)
    print(f"  Max Diff: {diff.max():.6f}")
    print(f"  Mean Diff: {diff.mean():.6f}")
    print("="*70 + "\n")


# ============================================================================
# Main Inference
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare Student vs Teacher Intermediate Results')
    
    parser.add_argument('--input_image', type=str, required=True,
                       help='Path to input low-light image')
    parser.add_argument('--teacher_path', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--student_path', type=str, required=True,
                       help='Path to student model checkpoint')
    parser.add_argument('--output_dir', type=str, default='image_comparison_results',
                       help='Directory to save results')
    parser.add_argument('--model_type', type=str, default='s',
                       help='Teacher model type')
    parser.add_argument('--normalize', action='store_true',
                       help='Apply normalization (usually False for low-light)')
    parser.add_argument('--gpu_id', type=str, default='0')
    
    args = parser.parse_args()
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("Student vs Teacher Comparison")
    print("="*70)
    print(f"Input image: {args.input_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # ==================== Load Models ====================
    print("Loading models...")
    
    # Teacher
    teacher = IAT(type=args.model_type, with_global=True).to(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher.eval()
    print(f"✅ Teacher loaded: {args.teacher_path}")
    
    # Student
    student = IAT_Student_BN(in_dim=3, dim=16).to(device)
    checkpoint = torch.load(args.student_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        student.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        psnr = checkpoint.get('best_psnr', 0)
        print(f"✅ Student loaded: {args.student_path}")
        print(f"   Epoch: {epoch}, Best PSNR: {psnr:.4f}")
    else:
        student.load_state_dict(checkpoint)
        print(f"✅ Student loaded: {args.student_path}")
    student.eval()
    
    # ==================== Load Image ====================
    print(f"\nLoading image: {args.input_image}")
    
    # Load image the same way as the original inference code
    img = np.asarray(Image.open(args.input_image), dtype=np.float32) / 255.0

    
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).float().to(device)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [H,W,C] -> [1,C,H,W]
    
    normalize_process = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img_tensor = normalize_process(img_tensor[0]).unsqueeze(0)
    
    print(f"  Image shape: {img_tensor.shape}")
    print(f"  Value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # ==================== Inference ====================
    print("\n" + "="*70)
    print("Running inference...")
    print("="*70)
    
    with torch.no_grad():
        # Student
        print("  Student model...")
        student_outputs = student(img_tensor, return_intermediates=True)
        student_results = {
            'input': img_tensor,
            'mul': student_outputs['mul'],
            'add': student_outputs['add'],
            'after_mul': img_tensor * student_outputs['mul'],
            'after_add': student_outputs['img_local'],
            'after_color': student_outputs['img_high'] ** (1 / student_outputs['gamma'].view(-1, 1, 1, 1)),  # gamma 적용 전 단계 복원
            'final': student_outputs['img_high'],
            'gamma': student_outputs['gamma'],
            'color': student_outputs['color']
        }
        
        # Teacher
        print("  Teacher model...")
        teacher_results = extract_teacher_intermediates(teacher, img_tensor)

    
    print("✅ Inference completed!\n")
    
    # ==================== Parameter Comparison ====================
    print_parameter_comparison(student_results, teacher_results)
    
    # ==================== Save Results ====================
    print("Saving results...")
    print("="*70)
    
    # 1. Comparison grid
    grid_path = os.path.join(args.output_dir, 'comparison_grid.png')
    save_comparison_grid(student_results, teacher_results, grid_path)
    
    # 2. Individual images - Student
    print("\nStudent individual images:")
    save_individual_images(student_results, 'student', args.output_dir)
    
    # 3. Individual images - Teacher
    print("\nTeacher individual images:")
    save_individual_images(teacher_results, 'teacher', args.output_dir)
    
    print("\n" + "="*70)
    print("All results saved!")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    # ==================== PSNR/SSIM Comparison ====================
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        
        student_final = tensor_to_image(student_results['final'])
        teacher_final = tensor_to_image(teacher_results['final'])
        
        psnr_val = psnr(teacher_final, student_final, data_range=1.0)
        ssim_val = ssim(teacher_final, student_final, 
                       data_range=1.0, channel_axis=2)
        
        print("\n" + "="*70)
        print("Quality Metrics (Student vs Teacher)")
        print("="*70)
        print(f"PSNR: {psnr_val:.4f} dB")
        print(f"SSIM: {ssim_val:.4f}")
        print("="*70 + "\n")
    except ImportError:
        print("\n⚠️  Install scikit-image for quality metrics: pip install scikit-image")


if __name__ == '__main__':
    main()

