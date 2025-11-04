import os
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from restormer.dataset import VimeoDataset  # Make sure this imports the updated version


def show_dataset_info(dataset_path):
    """Show detailed information about the dataset structure"""
    print("\n=== Dataset Structure Analysis ===")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path {dataset_path} does not exist!")
        return

    print(f"Dataset root: {dataset_path}")
    root_contents = os.listdir(dataset_path)
    print(f"Root contents: {root_contents[:10]}")

    # Check for Vimeo dataset structure
    vimeo_dir = os.path.join(dataset_path, "vimeo_settuplet_1")
    if os.path.exists(vimeo_dir) and os.path.isdir(vimeo_dir):
        print("Found vimeo_septuplet directory")
        vimeo_contents = os.listdir(vimeo_dir)
        print(f"Contents of vimeo_septuplet: {vimeo_contents[:10]}")

        sequences_dir = os.path.join(vimeo_dir, "sequences")
        if os.path.exists(sequences_dir) and os.path.isdir(sequences_dir):
            print(f"Found sequences directory at {sequences_dir}")
            seq_folders = [
                d
                for d in os.listdir(sequences_dir)
                if os.path.isdir(os.path.join(sequences_dir, d))
            ]
            print(f"Found {len(seq_folders)} sequence folders in sequences directory")
            if seq_folders:
                print(f"Example folders: {seq_folders[:5]}")

                # Check sample folder structure
                sample_folder = os.path.join(sequences_dir, seq_folders[0])
                sample_contents = os.listdir(sample_folder)
                print(f"Contents of {seq_folders[0]}: {sample_contents[:10]}")

                # Check if it contains subdirectories with sequences
                subdirs = [
                    d
                    for d in sample_contents
                    if os.path.isdir(os.path.join(sample_folder, d))
                ]
                if subdirs:
                    print(f"Found {len(subdirs)} subdirectories in {seq_folders[0]}")
                    sample_subdir = os.path.join(sample_folder, subdirs[0])
                    subdir_contents = os.listdir(sample_subdir)
                    print(f"Contents of {subdirs[0]}: {subdir_contents}")
                    
                    # Check if it contains image files
                    image_files = [f for f in subdir_contents if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if image_files:
                        print(f"Found {len(image_files)} image files in {subdirs[0]}")
        else:
            print("No sequences directory found, checking vimeo_septuplet directly")
            image_files = [f for f in os.listdir(vimeo_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                print(f"Found {len(image_files)} image files in vimeo_septuplet")

    # Check for split files
    for list_name in ["sep_trainlist.txt", "sep_testlist.txt", "test.txt"]:
        list_path = os.path.join(dataset_path, list_name)
        if os.path.exists(list_path):
            with open(list_path) as f:
                lines = f.readlines()
                print(f"Found {list_name} with {len(lines)} entries")
                if lines:
                    sample_entries = [line.strip() for line in lines[:3]]
                    print(f"Sample entries: {sample_entries}")
        else:
            print(f"Could not find {list_name}")

    print("=== End of Dataset Analysis ===\n")


def visualize_sample(dataset):
    """Visualize a random sample from the dataset"""
    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    idx = random.randint(0, len(dataset) - 1)
    lr_frames, hr_frame, bicubic_frame = dataset[idx]  # Updated for 3 outputs

    print("Sample shapes:")
    print(f"  LR frames: {lr_frames.shape}")  # [N, C, H, W]
    print(f"  HR frame: {hr_frame.shape}")    # [C, H, W]
    print(f"  Bicubic frame: {bicubic_frame.shape}")  # [C, H, W]

    # Convert to numpy for visualization
    lr_frames_np = lr_frames.numpy()
    hr_frame_np = hr_frame.numpy()
    bicubic_frame_np = bicubic_frame.numpy()

    # Transpose for matplotlib (CHW -> HWC)
    lr_vis = np.transpose(lr_frames_np, (0, 2, 3, 1))  # [N, H, W, C]
    hr_vis = np.transpose(hr_frame_np, (1, 2, 0))       # [H, W, C]
    bicubic_vis = np.transpose(bicubic_frame_np, (1, 2, 0))  # [H, W, C]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Show LR frames
    for i in range(3):
        axes[0, i].imshow(lr_vis[i])
        axes[0, i].set_title(f"LR Frame {i + 1}")
        axes[0, i].axis("off")

    axes[0, 3].imshow(bicubic_vis)
    axes[0, 3].set_title("Bicubic Upsampled")
    axes[0, 3].axis("off")

    # Show HR and comparisons
    axes[1, 0].imshow(hr_vis)
    axes[1, 0].set_title("HR Ground Truth")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(bicubic_vis)
    axes[1, 1].set_title("Bicubic (Reference)")
    axes[1, 1].axis("off")

    # Show difference
    diff = np.abs(hr_vis - bicubic_vis)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title("HR - Bicubic Difference")
    axes[1, 2].axis("off")

    # Show PSNR comparison
    axes[1, 3].text(0.1, 0.5, f"Bicubic PSNR: {calculate_psnr(hr_vis, bicubic_vis):.2f} dB\n"
                    f"Scale Factor: {dataset.scale_factor}\n"
                    f"Frame Count: {dataset.frame_count}", 
                    fontsize=12, transform=axes[1, 3].transAxes)
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig("sample_visualization.png", dpi=300, bbox_inches='tight')
    print("Sample visualization saved to 'sample_visualization.png'")


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def test_data_loading(dataset_path, split_list=None, batch_size=2, scale_factor=4, frame_count=3, patch_size=64, is_training=True):
    """Test loading the dataset with DataLoader"""
    try:
        print(f"Creating dataset with scale_factor={scale_factor}, frame_count={frame_count}, is_training={is_training}")
        
        dataset = VimeoDataset(
            root_dir=dataset_path,
            split_list=split_list,
            scale_factor=scale_factor,
            frame_count=frame_count,
            patch_size=patch_size if is_training else 128,
            is_training=is_training,
            verbose=True
        )

        print(f"Dataset size: {len(dataset)} sequences")

        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=is_training, 
            num_workers=2,
            pin_memory=True
        )

        print(f"Testing dataloader with {len(dataloader)} batches")

        num_batches_to_test = min(3, len(dataloader))
        for i, (lr_frames, hr_frames, bicubic_frames) in enumerate(dataloader):
            if i >= num_batches_to_test:
                break

            print(f"\nBatch {i + 1}:")
            print(f"  LR frames shape: {lr_frames.shape}")      # [B, N, C, H, W]
            print(f"  HR frames shape: {hr_frames.shape}")      # [B, C, H, W]
            print(f"  Bicubic frames shape: {bicubic_frames.shape}")  # [B, C, H, W]
            print(f"  LR range: [{lr_frames.min():.3f}, {lr_frames.max():.3f}]")
            print(f"  HR range: [{hr_frames.min():.3f}, {hr_frames.max():.3f}]")
            print(f"  Bicubic range: [{bicubic_frames.min():.3f}, {bicubic_frames.max():.3f}]")

        # Visualize a sample
        visualize_sample(dataset)

        print("\n✅ Dataset loading test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during dataset loading test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Default dataset path
    dataset_path = "/kaggle/input/archive"  # Updated common path
    
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if not arg_path.startswith("-"):  
            dataset_path = arg_path

    # Show dataset structure
    show_dataset_info(dataset_path)

    # Test with training data
    train_list = os.path.join(dataset_path, "sep_trainlist.txt")  
    if os.path.exists(train_list):
        print("\n Testing with training split list...")
        test_data_loading(dataset_path, train_list, batch_size=2, patch_size=64, is_training=True)
    else:
        print("\n Testing without split list (training mode)...")
        test_data_loading(dataset_path, batch_size=2, patch_size=64, is_training=True)

    # Test with validation data (larger patch size, no augmentation)
    test_list = os.path.join(dataset_path, "sep_testlist.txt")
    if os.path.exists(test_list):
        print("\n Testing with test split list (validation mode)...")
        test_data_loading(dataset_path, test_list, batch_size=1, patch_size=128, is_training=False)