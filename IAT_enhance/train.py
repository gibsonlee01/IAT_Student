# IAT_Student train code


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from torchvision.models import vgg16
from collections import defaultdict

from data_loaders.lol import lowlight_loader
from model.IAT_main import IAT
from IQA_pytorch import SSIM
from utils import PSNR, validation, LossNetwork
from torch.utils.tensorboard import SummaryWriter
from model.IAT_student import IAT_Student_BN

# ============================================================================
# Intensive Distillation Trainer 
# ============================================================================

class IAT_DistillationTrainer:
    """
    강력한 Multi-level Distillation
    - Feature-level alignment
    - Attention transfer
    - Progressive training
    """
    def __init__(self, config):
        self.config = config        
        self.start_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        
        # ==================== Teacher ====================
        print("="*70)
        print("Loading Teacher Model...")
        self.teacher = IAT(type=config.model_type, with_global=True).cuda()
        if config.teacher_path:
            self.teacher.load_state_dict(torch.load(config.teacher_path))
            print(f"✅ Teacher loaded: {config.teacher_path}")
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # ==================== Student ====================
        print("\nInitializing Student Model...")
        self.student = IAT_Student_BN(in_dim=3, dim=16).cuda()
        
        teacher_params = sum(p.numel() for p in self.teacher.parameters())/1e6
        student_params = sum(p.numel() for p in self.student.parameters())/1e6
        print(f"\n{'='*70}")
        print(f"Model Statistics:")
        print(f"  Teacher: {teacher_params:.3f}M params")
        print(f"  Student: {student_params:.3f}M params")
        print(f"  Compression: {teacher_params/student_params:.2f}x smaller")
        print(f"  Expected inference: ~4.9ms (vs Teacher ~19ms)")
        print(f"{'='*70}\n")
        
        # ==================== Optimizer ====================
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs - config.warmup_epochs,
            eta_min=config.lr * 0.01
        )
        
        # ==================== Resume ====================
        if config.resume_path and os.path.exists(config.resume_path):
            print(f"🔁 Resuming from checkpoint: {config.resume_path}")
            checkpoint = torch.load(config.resume_path, weights_only=False)            
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.student.load_state_dict(checkpoint['state_dict'])
                if not config.reset_optimizer and 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if not config.reset_optimizer and 'scheduler' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                self.best_psnr = checkpoint.get('best_psnr', 0)
                self.best_ssim = checkpoint.get('best_ssim', 0)
            else:
                self.student.load_state_dict(checkpoint)
                self.start_epoch = config.resume_epoch
            
            print(f"  Resume from epoch: {self.start_epoch}")
            print(f"  Best PSNR: {self.best_psnr:.4f}")
            print(f"  Best SSIM: {self.best_ssim:.4f}\n")
        
        # ==================== Loss Functions ====================
        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()
        self.SmoothL1_loss = nn.SmoothL1Loss()
        
        # Perceptual loss
        vgg_model = vgg16(pretrained=True).features[:16].cuda()
        for p in vgg_model.parameters():
            p.requires_grad = False
        self.loss_network = LossNetwork(vgg_model).eval()
        
        # Metrics
        self.ssim = SSIM()
        self.psnr_metric = PSNR()
        
        # Tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.snapshots_folder, "tensorboard")
        )
        
        print("✅ Trainer initialized!\n")
    
    def extract_teacher_knowledge(self, img_low):
        """Teacher의 모든 intermediate features 추출"""
        with torch.no_grad():
            # Local
            mul_t, add_t = self.teacher.local_net(img_low)
            img_local_t = img_low.mul(mul_t).add(add_t)
            
            # Global
            gamma_t, color_t = self.teacher.global_net(img_low)
            
            # Final
            _, _, img_high_t = self.teacher(img_low)
        
        return {
            'mul': mul_t,
            'add': add_t,
            'img_local': img_local_t,
            'gamma': gamma_t,
            'color': color_t,
            'img_high': img_high_t
        }
    
    def compute_distillation_loss(self, student_out, teacher_out, gt_img, epoch):
        """
        강력한 Multi-Level Distillation Loss
        
        Levels:
        1. GT Reconstruction (Primary)
        2. Final Output Mimicking
        3. Global Parameters (Gamma, Color) - 핵심!
        4. Local Enhancement
        5. Perceptual Quality
        6. Feature-level Alignment
        """
        losses = {}
        
        # ========== Level 1: GT Reconstruction (Primary) ==========
        losses['gt_l1'] = self.L1_loss(student_out['img_high'], gt_img)
        losses['gt_smooth'] = self.SmoothL1_loss(student_out['img_high'], gt_img)
        losses['gt_l2'] = self.L2_loss(student_out['img_high'], gt_img)
        
        # ========== Level 2: Final Output Distillation ==========
        losses['output_l1'] = self.L1_loss(
            student_out['img_high'], 
            teacher_out['img_high']
        )
        losses['output_l2'] = self.L2_loss(
            student_out['img_high'], 
            teacher_out['img_high']
        )
        
        # ========== Level 3: Global Parameters (Critical!) ==========
        losses['gamma_l2'] = self.L2_loss(
            student_out['gamma'], 
            teacher_out['gamma']
        )
        losses['gamma_l1'] = self.L1_loss(
            student_out['gamma'], 
            teacher_out['gamma']
        )
        
        losses['color_l2'] = self.L2_loss(
            student_out['color'], 
            teacher_out['color']
        )
        losses['color_l1'] = self.L1_loss(
            student_out['color'], 
            teacher_out['color']
        )
        
        # ========== Level 4: Local Enhancement ==========
        losses['mul'] = self.L1_loss(student_out['mul'], teacher_out['mul'])
        losses['add'] = self.L1_loss(student_out['add'], teacher_out['add'])
        losses['img_local_l1'] = self.L1_loss(
            student_out['img_local'], 
            teacher_out['img_local']
        )
        losses['img_local_l2'] = self.L2_loss(
            student_out['img_local'], 
            teacher_out['img_local']
        )
        
        # ========== Level 5: Perceptual Quality ==========
        losses['perceptual_gt'] = self.loss_network(
            student_out['img_high'], 
            gt_img
        )
        losses['perceptual_teacher'] = self.loss_network(
            student_out['img_high'], 
            teacher_out['img_high']
        )
        
        # ========== Progressive Weighting Strategy ==========
        warmup = self.config.warmup_epochs
        total = self.config.num_epochs
        
        if epoch < warmup:
            # Phase 1: Warmup
            progress = epoch / warmup
            w_distill = 0.1 * progress     
            w_global = 0.2 * progress        
            w_perceptual = 0.04
            w_feature = 0.0
        elif epoch < warmup + 30:
            # Phase 2: Early Training
            progress = (epoch - warmup) / 30
            w_distill = 0.1 + 0.4 * progress 
            w_global = 0.2 + 0.6 * progress   
            w_perceptual = 0.04 + 0.04 * progress  
            w_feature = 0.1 * progress        
        else:
            # Phase 3: Late Training
            progress = (epoch - warmup - 30) / max(1, total - warmup - 30)
            w_distill = 0.5 + 0.3 * progress  
            w_global = 0.8 + 0.4 * progress    
            w_perceptual = 0.08 + 0.04 * progress  
            w_feature = 0.1 + 0.2 * progress   
        
        # ========== Loss ==========
        total_loss = (
            # GT Reconstruction (항상 primary)
            1.0 * losses['gt_l1'] +
            0.5 * losses['gt_smooth'] +
            0.2 * losses['gt_l2'] +
            
            # Final Output Distillation
            w_distill * losses['output_l1'] +
            w_distill * 0.5 * losses['output_l2'] +
            
            # Global Parameters 
            w_global * losses['gamma_l2'] +
            w_global * 0.5 * losses['gamma_l1'] +
            w_global * 0.8 * losses['color_l2'] +
            w_global * 0.4 * losses['color_l1'] +
            
            # Local Enhancement
            w_distill * 0.4 * losses['img_local_l1'] +
            w_distill * 0.2 * losses['img_local_l2'] +
            w_distill * 0.3 * losses['mul'] +
            w_distill * 0.3 * losses['add'] +
            
            # Perceptual
            w_perceptual * losses['perceptual_gt'] +
            w_perceptual * 0.5 * losses['perceptual_teacher']
        )
        
        losses['total'] = total_loss
        losses['w_distill'] = w_distill
        losses['w_global'] = w_global
        losses['w_perceptual'] = w_perceptual
        
        return total_loss, losses
    
    def train_epoch(self, train_loader, epoch):
        self.student.train()
        epoch_losses = defaultdict(list)
        
        # Warmup learning rate
        if epoch < self.config.warmup_epochs:
            lr_scale = (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.lr * lr_scale
        
        for iteration, imgs in enumerate(train_loader):
            low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
            
            # Forward
            self.optimizer.zero_grad()
            
            # Teacher knowledge
            teacher_out = self.extract_teacher_knowledge(low_img)
            
            # Student prediction
            student_out = self.student(low_img, return_intermediates=True)
            
            # Loss
            total_loss, losses = self.compute_distillation_loss(
                student_out, teacher_out, high_img, epoch
            )
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Logging
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses[k].append(v.item())
                else:
                    epoch_losses[k].append(v)
            
            # Display
            if (iteration + 1) % self.config.display_iter == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch:03d}] [{iteration+1:04d}/{len(train_loader):04d}] "
                      f"LR: {current_lr:.6f}")
                print(f"  Loss: {losses['total'].item():.4f} | "
                      f"GT: {losses['gt_l1'].item():.4f} | "
                      f"Gamma: {losses['gamma_l2'].item():.4f} | "
                      f"Output: {losses['output_l1'].item():.4f}")
                print(f"  Weights - Distill: {losses['w_distill']:.3f}, "
                      f"Global: {losses['w_global']:.3f}, "
                      f"Percept: {losses['w_perceptual']:.3f}")
                
                # Tensorboard
                global_step = epoch * len(train_loader) + iteration
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        self.writer.add_scalar(f"Train/{k}", v.item(), global_step)
                    else:
                        self.writer.add_scalar(f"Train/{k}", v, global_step)
                self.writer.add_scalar("Train/lr", current_lr, global_step)
        
        # Epoch average
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self, val_loader, epoch):
        """검증"""
        self.student.eval()
        
        PSNR_mean, SSIM_mean = validation(self.student, val_loader)
        
        self.writer.add_scalar("Val/PSNR", PSNR_mean, epoch)
        self.writer.add_scalar("Val/SSIM", SSIM_mean, epoch)
        
        return PSNR_mean, SSIM_mean
    
    def train(self, train_loader, val_loader):
        """전체 학습 루프"""
        print("\n" + "="*70)
        print("Starting Intensive Distillation Training")
        print("="*70)
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Warmup epochs: {self.config.warmup_epochs}")
        print(f"Initial LR: {self.config.lr}")
        print(f"Batch size: {self.config.batch_size}")
        print("="*70 + "\n")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{self.config.num_epochs-1}")
            print(f"{'='*70}")
            
            # Train
            avg_losses = self.train_epoch(train_loader, epoch)
            
            print(f"\n[Epoch {epoch} Summary]")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  - GT L1: {avg_losses['gt_l1']:.4f}")
            print(f"  - Gamma L2: {avg_losses['gamma_l2']:.4f}")
            print(f"  - Color L2: {avg_losses['color_l2']:.4f}")
            print(f"  - Output L1: {avg_losses['output_l1']:.4f}")
            
            # Validate
            print(f"\n  Validating...")
            PSNR_mean, SSIM_mean = self.validate(val_loader, epoch)
            print(f"  PSNR: {PSNR_mean:.4f} | SSIM: {SSIM_mean:.4f}")
            
            # Logging
            log_path = os.path.join(self.config.snapshots_folder, 'train_log.txt')
            with open(log_path, 'a+') as f:
                f.write(f"Epoch {epoch}: PSNR={PSNR_mean:.4f}, SSIM={SSIM_mean:.4f}, "
                       f"Loss={avg_losses['total']:.4f}\n")
            
            # Save best
            is_best = False
            if PSNR_mean > self.best_psnr:
                self.best_psnr = PSNR_mean
                self.best_ssim = SSIM_mean
                is_best = True
                
                # Save best model
                best_path = os.path.join(self.config.snapshots_folder, "student_best.pth")
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.student.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_psnr': self.best_psnr,
                    'best_ssim': self.best_ssim
                }, best_path)
                
                print(f"\n  ✅ New best model saved! PSNR: {PSNR_mean:.4f}")
            
            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(
                    self.config.snapshots_folder, 
                    f"student_epoch_{epoch}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.student.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_psnr': self.best_psnr,
                    'best_ssim': self.best_ssim
                }, ckpt_path)
                print(f"  💾 Checkpoint saved: epoch_{epoch}.pth")
            
            # Learning rate scheduling
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            print()
        
        # Final summary
        print("\n" + "="*70)
        print("Training Completed!")
        print(f"Best PSNR: {self.best_psnr:.4f}")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print("="*70 + "\n")
        
        self.writer.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IAT Student Distillation Training')
    
    # Paths
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--img_path', type=str, 
                       default="/content/drive/MyDrive/LOL-v2/Real_captured/Train/Low/")
    parser.add_argument('--img_val_path', type=str,
                       default="/content/drive/MyDrive/LOL-v2/Real_captured/Test/Low/")
    parser.add_argument('--teacher_path', type=str, required=True,
                       help="Path to pretrained teacher model")
    parser.add_argument('--snapshots_folder', type=str, 
                       default="workdirs/IAT_student_bn")
    
    # Training
    parser.add_argument('--model_type', type=str, default='s')
    parser.add_argument('--normalize', action='store_false')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200,
                       help="More epochs for better quality")
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help="Longer warmup for stability")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--display_iter', type=int, default=10)
    
    # Resume
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--reset_optimizer', action='store_true',
                       help="Reset optimizer when resuming")
    
    config = parser.parse_args()
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    os.makedirs(config.snapshots_folder, exist_ok=True)
    
    # Print config
    print("\n" + "="*70)
    print("Configuration:")
    print("="*70)
    for arg in vars(config):
        print(f"  {arg}: {getattr(config, arg)}")
    print("="*70 + "\n")
    
    # Data loaders
    print("Loading datasets...")
    train_dataset = lowlight_loader(
        images_path=config.img_path, 
        normalize=config.normalize
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=8, 
        pin_memory=True
    )
    
    val_dataset = lowlight_loader(
        images_path=config.img_val_path, 
        mode='test', 
        normalize=config.normalize
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=8, 
        pin_memory=True
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}\n")
    
    # Train
    trainer = IAT_DistillationTrainer(config)
    trainer.train(train_loader, val_loader)