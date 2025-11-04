import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from restormer.models.restormer_arch import Restormer 

class RestormerTeacher(nn.Module):
    """
    Teacher model for video super-resolution knowledge distillation, using Restormer.
    Enhanced for better compatibility with official Restormer pretrained weights.
    """
    def __init__(self, checkpoint_path=None, scale_factor=1, device='cpu'):
        super().__init__()
        
        self.device = torch.device(device)
        self.scale_factor = scale_factor
        
        print(f"🔧 Initializing RestormerTeacher on device: {self.device}")
        
        # Use exact architecture that matches pretrained weights
        self.restormer = Restormer(
            inp_channels=3, 
            out_channels=3, 
            dim=48,
            num_blocks=[4,6,6,8], 
            num_refinement_blocks=4,
            heads=[1,2,4,8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False
        )
        
        # Move the entire model to the specified device
        self.restormer = self.restormer.to(self.device)
        
        if checkpoint_path is not None:
            self.load_pretrained_weights(checkpoint_path)
        else:
            print("🔄 No checkpoint provided, using randomly initialized weights")
        
        # Freeze teacher
        self.restormer.eval()
        for param in self.restormer.parameters():
            param.requires_grad = False
            
        print(f"✅ RestormerTeacher initialized on {self.device}")

    def load_pretrained_weights(self, checkpoint_path):
        """Enhanced loading for official Restormer weights"""
        try:
            if not os.path.exists(checkpoint_path):
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                return False
                
            print(f"📂 Loading pretrained weights from: {checkpoint_path}")
            print(f"📦 Loading to device: {self.device}")
            
            # Load checkpoint directly to the target device
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            
            # Official Restormer weights usually have 'params' key
            if 'params' in ckpt:
                state_dict = ckpt['params']
                print("✅ Found 'params' key in checkpoint")
            elif 'model' in ckpt:
                state_dict = ckpt['model']
                print("✅ Found 'model' key in checkpoint")
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict'] 
                print("✅ Found 'state_dict' key in checkpoint")
            else:
                state_dict = ckpt
                print("✅ Using direct state_dict")
            
            # Clean keys (remove any prefix)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith('module.'):
                    new_key = k[7:]  # remove 'module.' prefix
                elif k.startswith('net.'):
                    new_key = k[4:]  # remove 'net.' prefix
                cleaned_state_dict[new_key] = v
            
            # Load with detailed error reporting
            load_result = self.restormer.load_state_dict(cleaned_state_dict, strict=False)
            
            print(f"📊 Loading Report:")
            print(f"   Missing keys: {len(load_result.missing_keys)}")
            print(f"   Unexpected keys: {len(load_result.unexpected_keys)}")
            
            if not load_result.missing_keys and not load_result.unexpected_keys:
                print("🎉 Perfect match! All weights loaded successfully.")
            else:
                if load_result.missing_keys:
                    print(f"⚠  First 5 missing keys: {load_result.missing_keys[:5]}")
                if load_result.unexpected_keys:
                    print(f"⚠  First 5 unexpected keys: {load_result.unexpected_keys[:5]}")
            
            # Verify all parameters are on the correct device
            all_on_correct_device = all(p.device == self.device for p in self.restormer.parameters())
            print(f"🔍 All parameters on {self.device}: {all_on_correct_device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Continuing with randomly initialized weights")
            return False

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 3, H, W] (single RGB frame)
        Returns:
            Teacher output tensor [B, 3, H', W'] 
        """
        with torch.no_grad():
            # Ensure input is on the same device as model
            if x.device != self.device:
                x = x.to(self.device)
                
            # Restormer expects input in range [0,1]
            if x.max() > 1.0:
                x = x / 255.0
                
            out = self.restormer(x)
            
            # Optional upscaling for super-resolution
            if self.scale_factor != 1:
                _, _, H, W = x.shape
                out = F.interpolate(out, size=(H * self.scale_factor, W * self.scale_factor),
                                    mode='bilinear', align_corners=False)
            
            # Convert back to original range if needed
            if x.max() > 1.0:
                out = out * 255.0
                
        return out

    def to(self, device):
        """Override to method to handle device changes properly"""
        self.device = torch.device(device)
        self.restormer = self.restormer.to(self.device)
        return self