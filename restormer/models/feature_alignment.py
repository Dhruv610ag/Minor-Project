import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAlignmentModule(nn.Module):
    """
    Feature alignment module for knowledge distillation between Restormer teacher and GhostNet student.
    Handles both teacher and student feature alignment and provides distillation hooks.
    """
    
    def __init__(self, alignment_type='concat', flow_estimation=False,
                 teacher_channels=None, student_channels=None, flow_scale=0.1):
        super().__init__()
        self.alignment_type = alignment_type
        self.flow_estimation = flow_estimation
        self.flow_scale = flow_scale

        if alignment_type == 'conv':
            # Example: student expects concatenated 3 frames -> 9 channels; adapt as needed.
            self.student_align = nn.Conv2d(9, 32, 3, padding=1)
            self.activation = nn.ReLU(inplace=True)
        else:
            self.student_align = None
            self.activation = nn.Identity()

        if flow_estimation:
            # Simple flow estimation (optional)
            # flow_conv expects concatenated center(3) + other(3) = 6 channels
            self.flow_conv = nn.Conv2d(6, 2, 3, padding=1)
        else:
            self.flow_conv = None

        # optional 1x1 conv to align teacher -> student channels for distillation
        self.align_conv = None
        if teacher_channels is not None and student_channels is not None and teacher_channels != student_channels:
            self.align_conv = nn.Conv2d(teacher_channels, student_channels, kernel_size=1)

    
    def forward_teacher(self, frames):
        """
        Align frames for teacher model (Restormer)
        Args:
            frames: [B, N, C, H, W] - multiple frames
        Returns:
            center_frame: [B, C, H, W] - center frame for teacher processing
        """
        # Teacher processes center frame only
        return frames[:, frames.size(1) // 2, :, :, :]
    
    def forward_student(self, frames):
        """
        Align frames for student model (GhostNet)
        Args:
            frames: [B, N, C, H, W] - multiple frames
        Returns:
            aligned_features: [B, channels, H, W] - aligned features for student
        """
        B, N, C, H, W = frames.shape

        if self.alignment_type == 'concat':
            return frames.view(B, N * C, H, W)

        elif self.alignment_type == 'conv' and self.student_align is not None:
            concatenated = frames.view(B, N * C, H, W)
            return self.activation(self.student_align(concatenated))

        elif self.alignment_type == 'flow' and self.flow_estimation and self.flow_conv is not None:
            center_frame = frames[:, N // 2, :, :, :]
            aligned_frames = []
            for i in range(N):
                if i == N // 2:
                    aligned_frames.append(center_frame)
                else:
                    flow_input = torch.cat([center_frame, frames[:, i, :, :, :]], dim=1)
                    flow = self.flow_conv(flow_input)
                    grid = self._create_flow_grid(flow, H, W)
                    warped = F.grid_sample(frames[:, i, :, :, :], grid, align_corners=False)
                    aligned_frames.append(warped)
            return torch.cat(aligned_frames, dim=1)

        return frames.view(B, N * C, H, W)
    
    def _create_flow_grid(self, flow, H, W):
        """
        Create sampling grid from optical flow
        """
        B, _, Hf, Wf = flow.shape
        # Create meshgrid with indexing='xy' for (x,y)
        grid_y, grid_x = torch.meshgrid(torch.arange(Hf, device=flow.device),
                                        torch.arange(Wf, device=flow.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).float().unsqueeze(0).repeat(B, 1, 1, 1)
        # normalize to [-1,1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (Wf - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (Hf - 1) - 1.0
        grid = grid + flow.permute(0, 2, 3, 1) * self.flow_scale
        return grid
    
    def get_distillation_features(self, teacher_features, student_features):
        """
        Prepare features for distillation loss calculation
        Args:
            teacher_features: Features from teacher model
            student_features: Features from student model
        Returns:
            aligned_teacher: Aligned teacher features for distillation
            aligned_student: Aligned student features for distillation
        """
        # Resize teacher spatial dims to match student if needed
        if teacher_features.shape[2:] != student_features.shape[2:]:
            teacher_features = F.adaptive_avg_pool2d(teacher_features, student_features.shape[2:])
        # Create / register adapter conv on first use (so channels are known)
        if teacher_features.shape[1] != student_features.shape[1]:
            if self.align_conv is None:
                self.align_conv = nn.Conv2d(teacher_features.shape[1], student_features.shape[1], 1).to(teacher_features.device)
                # ensure align_conv is registered as a module
                setattr(self, "align_conv", self.align_conv)
            teacher_features = self.align_conv(teacher_features)
        return teacher_features, student_features