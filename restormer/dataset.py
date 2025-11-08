import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class VimeoDataset(Dataset):
    """
    Dataset for loading sequences of frames from the Vimeo-90k dataset.
    Returns (input frames, target frame) for image enhancement.
    """
    def __init__(self, root_dir, split_list=None, sample_size=None,
                 scale_factor=1, frame_count=3, patch_size=64,  # CHANGED: scale_factor=1
                 is_training=True, verbose=True):
        self.root_dir = root_dir
        self.scale_factor = scale_factor  # Now 1 for same resolution
        self.frame_count = frame_count
        self.patch_size = patch_size
        self.is_training = is_training
        self.verbose = verbose
        self.sequences = []
        self.file_pattern = None

        # Detect sequence directory
        seq_dir = self._find_sequence_directory(root_dir)
        self.seq_dir = seq_dir

        # Load sequences from split list or scan directory
        if split_list is not None and os.path.exists(split_list):
            self._load_from_split_list(split_list, seq_dir)
        else:
            self._scan_directory_structure(seq_dir)

        if verbose:
            print(f"Found {len(self.sequences)} valid sequences")

        # Random sampling if needed
        if sample_size is not None and sample_size < len(self.sequences):
            if verbose:
                print(f"Using random subset of {sample_size} sequences")
            self.sequences = random.sample(self.sequences, sample_size)
    
    def _find_sequence_directory(self, root_dir):
        """Find the directory containing video sequences."""
        if self.verbose:
            print(f"Searching for sequences in: {root_dir}")
            if os.path.exists(root_dir):
                contents = os.listdir(root_dir)
                print(f"Root directory contents: {contents[:10]}...")
            else:
                print(f"Root directory does not exist: {self.root_dir}")

        # Common Vimeo structure
        vimeo_septuplet = os.path.join(root_dir, "vimeo_septuplet", "sequences")
        if os.path.exists(vimeo_septuplet):
            if self.verbose:
                print(f"Found sequences directory: {vimeo_septuplet}")
            return vimeo_septuplet

        vimeo_settuplet_1 = os.path.join(root_dir, "vimeo_settuplet_1")
        if os.path.exists(vimeo_settuplet_1):
            sequences_dir = os.path.join(vimeo_settuplet_1, "sequences")
            if os.path.exists(sequences_dir):
                if self.verbose:
                    print(f"Found sequences directory: {sequences_dir}")
                return sequences_dir
            return vimeo_settuplet_1

        # Try generic names
        for dirname in ["sequence", "sequences"]:
            candidate = os.path.join(root_dir, dirname)
            if os.path.isdir(candidate):
                if self.verbose:
                    print(f"Found {dirname} directory: {candidate}")
                return candidate

        # Check other potential dirs
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path) and self._has_video_structure(subdir_path):
                if self.verbose:
                    print(f"Found potential sequence directory: {subdir_path}")
                return subdir_path

        if self.verbose:
            print(f"Using root directory as fallback: {root_dir}")
        return root_dir

    def _has_video_structure(self, directory):
        """Check if directory has sequence-like structure."""
        try:
            contents = os.listdir(directory)
            has_dirs = any(item.isdigit() for item in contents if os.path.isdir(os.path.join(directory, item)))
            has_imgs = any(item.lower().endswith(('.png', '.jpg', '.jpeg')) for item in contents)
            return has_dirs or has_imgs
        except:
            return False

    def _load_from_split_list(self, split_list, seq_dir, max_sequences=None):
        """Load sequences from a text split list."""
        with open(split_list, 'r') as f:
            lines = f.readlines()
        if max_sequences:
            lines = lines[:max_sequences]
            if self.verbose:
                print(f"Limiting to {max_sequences} sequences from split list")

        for line in lines:
            seq = line.strip()
            seq_path = os.path.join(seq_dir, seq)
            if os.path.exists(seq_path) and self._check_sequence_valid(seq_path):
                self.sequences.append(seq_path)

    def _scan_directory_structure(self, seq_dir, max_sequences=None):
        """Scan directory structure for valid sequences."""
        if self._check_sequence_valid(seq_dir):
            self.sequences.append(seq_dir)
            return

        try:
            for root, dirs, _ in os.walk(seq_dir):
                for d in dirs:
                    if max_sequences and len(self.sequences) >= max_sequences:
                        return
                    d_path = os.path.join(root, d)
                    if self._check_sequence_valid(d_path):
                        self.sequences.append(d_path)
        except Exception as e:
            if self.verbose:
                print(f"Error scanning {seq_dir}: {e}")

    def _check_sequence_valid(self, seq_path):
        """Verify if sequence folder contains expected frames."""
        patterns = [
            [f'im{i}.png' for i in range(1, 4)],
            [f'im{i:02d}.png' for i in range(1, 4)],
            [f'frame{i:03d}.png' for i in range(1, 4)],
            [f'{i:02d}.png' for i in range(1, 4)]
        ]

        for pattern in patterns:
            valid = all(os.path.isfile(os.path.join(seq_path, f)) for f in pattern)
            if valid:
                self.file_pattern = pattern
                return True
        return False
    
    def _random_crop(self, frames, target_frame):
        """Perform random crop on input and target frames (same resolution)"""
        h, w = frames[0].shape[:2]
        crop_x = random.randint(0, w - self.patch_size)
        crop_y = random.randint(0, h - self.patch_size)
        
        cropped_frames = [f[crop_y:crop_y + self.patch_size, crop_x:crop_x + self.patch_size] for f in frames]
        cropped_target = target_frame[crop_y:crop_y + self.patch_size, crop_x:crop_x + self.patch_size]
        
        return cropped_frames, cropped_target

    def _create_dummy_data(self):
        """Return dummy data if dataset loading fails."""
        dummy_lr = torch.zeros(self.frame_count, 3, self.patch_size, self.patch_size)
        dummy_hr = torch.zeros(3, self.patch_size * self.scale_factor, self.patch_size * self.scale_factor)
        dummy_bicubic = dummy_hr.clone()
        return dummy_lr, dummy_hr, dummy_bicubic

    def __len__(self):
        return len(self.sequences) if self.sequences else 1

    def __getitem__(self, idx):
        if len(self.sequences) == 0:
            return self._create_dummy_data()

        idx = idx % len(self.sequences)
        seq_path = self.sequences[idx]

        try:
            # Load frames
            frames = []
            for frame_name in self.file_pattern[:self.frame_count]:
                frame_path = os.path.join(seq_path, frame_name)
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError(f"Failed to read image: {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            # Get center frame as target
            center_idx = len(frames) // 2
            target_frame = frames[center_idx].copy()

            # CHANGED: No downsampling - use original frames as input
            input_frames = frames  # Same resolution as target

            # CHANGED: bicubic is now the center input frame (same resolution)
            bicubic_frame = frames[center_idx].copy()

            # Data augmentation for training
            if self.is_training and self.patch_size > 0:
                # Ensure image is large enough for crop
                h, w = input_frames[0].shape[:2]
                if w < self.patch_size or h < self.patch_size:
                    # fallback: resize so we can crop
                    scale_up = max(self.patch_size / max(w, h), 1.0)
                    new_w = int(w * scale_up) + 1
                    new_h = int(h * scale_up) + 1
                    input_frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in input_frames]
                    target_frame = cv2.resize(target_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    bicubic_frame = cv2.resize(bicubic_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                input_frames, target_frame = self._random_crop(input_frames, target_frame)
                # Update bicubic to match cropped target
                bicubic_frame = input_frames[center_idx]

            # Convert to torch tensors
            input_tensor = torch.from_numpy(np.stack(input_frames, axis=0))  # [N, H, W, C]
            input_tensor = input_tensor.permute(0, 3, 1, 2).float()  # [N, C, H, W]

            target_tensor = torch.from_numpy(target_frame).permute(2, 0, 1).float()  # [C, H, W]
            bicubic_tensor = torch.from_numpy(bicubic_frame).permute(2, 0, 1).float()  # [C, H, W]

            return input_tensor, target_tensor, bicubic_tensor

        except Exception as e:
            if self.verbose:
                print(f"Error loading sequence {seq_path}: {e}")
            return self._create_dummy_data()