import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBlockDistillation(nn.Module):
    """
    Multi-Block Distillation (MBD) for knowledge distillation between
    Restormer teacher and GhostNet student.
    
    Distills knowledge from multiple blocks/levels of the teacher model
    to corresponding blocks in the student model.
    """
    
    def __init__(self, distillation_layers=None, loss_type='l1', temperature=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        
        # Default distillation layers if not specified
        if distillation_layers is None:
            self.distillation_layers = ['encoder1', 'encoder2', 'encoder3', 'bottleneck']
        else:
            self.distillation_layers = distillation_layers
        
        # Feature adapters for dimension matching
        self.feature_adapters = nn.ModuleDict()
    
    def create_adapters(self, teacher_feature_shapes, student_feature_shapes):
        """
        Create 1x1 conv adapters to match feature dimensions between teacher and student
        """
        for layer_name in self.distillation_layers:
            if layer_name in teacher_feature_shapes and layer_name in student_feature_shapes:
                t_channels = teacher_feature_shapes[layer_name][1]
                s_channels = student_feature_shapes[layer_name][1]
                
                if t_channels != s_channels:
                    self.feature_adapters[layer_name] = nn.Conv2d(t_channels, s_channels, 1)
    
    def forward(self, teacher_features, student_features):
        """
        Calculate distillation loss between teacher and student features
        
        Args:
            teacher_features: Dict of features from teacher model {layer_name: tensor}
            student_features: Dict of features from student model {layer_name: tensor}
        
        Returns:
            distillation_loss: Total distillation loss
        """
        total_loss = 0.0
        loss_dict = {}
        
        for layer_name in self.distillation_layers:
            if layer_name in teacher_features and layer_name in student_features:
                t_feat = teacher_features[layer_name]
                s_feat = student_features[layer_name]
                
                # Adapt teacher features to match student dimensions if needed
                if layer_name in self.feature_adapters:
                    t_feat = self.feature_adapters[layer_name](t_feat)
                
                # Ensure spatial dimensions match
                if t_feat.shape[2:] != s_feat.shape[2:]:
                    t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[2:])
                
                # Calculate distillation loss
                layer_loss = self.calculate_distillation_loss(t_feat, s_feat)
                total_loss += layer_loss
                loss_dict[f'distill_{layer_name}'] = layer_loss
        
        return total_loss, loss_dict
    
    def calculate_distillation_loss(self, teacher_feat, student_feat):
        """
        Calculate distillation loss between features
        """
        if self.loss_type == 'l1':
            return F.l1_loss(student_feat, teacher_feat.detach())
        
        elif self.loss_type == 'l2':
            return F.mse_loss(student_feat, teacher_feat.detach())
        
        elif self.loss_type == 'kl':
            # KL divergence for feature distillation
            soft_teacher = F.softmax(teacher_feat.detach() / self.temperature, dim=1)
            soft_student = F.log_softmax(student_feat / self.temperature, dim=1)
            return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        elif self.loss_type == 'attention':
            # Attention transfer loss
            return self.attention_transfer_loss(teacher_feat, student_feat)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def attention_transfer_loss(self, teacher_feat, student_feat):
        """
        Attention Transfer loss from "Paying More Attention to Attention"
        """
        # Calculate attention maps
        teacher_attention = self._get_attention_map(teacher_feat)
        student_attention = self._get_attention_map(student_feat)
        
        return F.mse_loss(student_attention, teacher_attention.detach())
    
    def _get_attention_map(self, features):
        """
        Calculate attention map from features
        """
        # Simple attention: sum of absolute values along channel dimension
        return torch.sum(torch.abs(features), dim=1, keepdim=True)
    
    def get_feature_hooks(self, model, is_teacher=False):
        """
        Create forward hooks to extract features from specific layers
        """
        hooks = {}
        features = {}
        
        def hook_fn(layer_name):
            def hook(module, input, output):
                features[layer_name] = output
            return hook
        
        # Register hooks based on model type
        if is_teacher:
            # Restormer teacher hooks
            if hasattr(model, 'encoder_level1'):
                hooks['encoder1'] = model.encoder_level1.register_forward_hook(hook_fn('encoder1'))
            if hasattr(model, 'encoder_level2'):
                hooks['encoder2'] = model.encoder_level2.register_forward_hook(hook_fn('encoder2'))
            if hasattr(model, 'encoder_level3'):
                hooks['encoder3'] = model.encoder_level3.register_forward_hook(hook_fn('encoder3'))
            if hasattr(model, 'latent'):
                hooks['bottleneck'] = model.latent.register_forward_hook(hook_fn('bottleneck'))
        else:
            # GhostNet student hooks
            if hasattr(model, 'bottleneck1'):
                hooks['encoder1'] = model.bottleneck1.register_forward_hook(hook_fn('encoder1'))
            if hasattr(model, 'bottleneck2'):
                hooks['encoder2'] = model.bottleneck2.register_forward_hook(hook_fn('encoder2'))
            if hasattr(model, 'bottleneck3'):
                hooks['bottleneck'] = model.bottleneck3.register_forward_hook(hook_fn('bottleneck'))
        
        return hooks, features