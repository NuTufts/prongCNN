#!/usr/bin/env python3
"""
Test script for ProngDatasetMultiFile to verify it works correctly
with PyTorch's DataLoader and multiprocessing.
"""

import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
import numpy as np
from collections import Counter

# Add models directory to path
sys.path.append('models')
from datasets_reco_5ClassHardLabel_quadTask_multifile import ProngDatasetMultiFile, mean, std

def test_single_item_access(dataset):
    """Test accessing individual items."""
    print("\n=== Testing single item access ===")
    
    # Test first, middle, and last items
    test_indices = [0, len(dataset)//2, len(dataset)-1]
    
    for idx in test_indices:
        print(f"\nTesting index {idx}...")
        try:
            image, target = dataset[idx]
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
            print(f"  Target: class={target[0]}, completeness={target[1]:.3f}, "
                  f"purity={target[2]:.3f}, process={target[3]}")
            
            # Check image dimensions
            assert image.shape == (6, 512, 512), f"Wrong image shape: {image.shape}"
            assert isinstance(target, list) and len(target) == 4, f"Wrong target format: {target}"
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return False
    
    return True

def test_dataloader(dataset, num_workers=4, batch_size=8, num_batches=50):
    """Test DataLoader with multiprocessing."""
    print(f"\n=== Testing DataLoader with {num_workers} workers ===")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        #pin_memory=True,
        #timeout=0
    )
    
    # Test loading batches
    print(f"Loading {len(dataloader)} batches...")
    start_time = time.time()
    
    class_counts = Counter()
    process_counts = Counter()
    completeness_values = []
    purity_values = []

    data_iter = iter(dataloader)
    
    for i in range( num_batches ):
        #, (images, targets) in enumerate(dataloader):
        #if i >= num_batches:  # Test first 10 batches
        #    break
        images,targets = next(data_iter)
            
        # Unpack targets (they come as list of tensors)
        classes = targets[0]
        completeness = targets[1]
        purity = targets[2]
        processes = targets[3]
        
        # Check batch dimensions
        assert images.shape[0] == batch_size or i == len(dataloader)-1
        assert images.shape[1:] == (6, 512, 512)
        
        # Collect statistics
        for c in classes.numpy():
            class_counts[c] += 1
        for p in processes.numpy():
            process_counts[p] += 1
        completeness_values.extend(completeness.numpy())
        purity_values.extend(purity.numpy())
        
        if i % 5 == 0:
            print(f"  Batch {i}: shape={images.shape}, "
                  f"time={time.time()-start_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nLoaded {min(10, len(dataloader))} batches in {total_time:.2f} seconds")
    print(f"Average time per batch: {total_time/min(10, len(dataloader)):.3f}s")
    
    # Print statistics
    print("\nClass distribution in loaded batches:")
    class_names = {0: "electron", 1: "photon", 2: "muon", 3: "pion", 4: "proton"}
    for cls, count in sorted(class_counts.items()):
        print(f"  {class_names.get(cls, 'unknown')}: {count}")
    
    print("\nProcess distribution:")
    process_names = {0: "primary", 1: "secondary_neutral", 2: "secondary_charged"}
    for proc, count in sorted(process_counts.items()):
        print(f"  {process_names.get(proc, 'unknown')}: {count}")
    
    print(f"\nCompleteness: mean={np.mean(completeness_values):.3f}, "
          f"std={np.std(completeness_values):.3f}")
    print(f"Purity: mean={np.mean(purity_values):.3f}, "
          f"std={np.std(purity_values):.3f}")
    
    return True

def test_file_distribution(dataset):
    """Test that sampling is properly distributed across files."""
    print("\n=== Testing file distribution ===")
    
    # Sample some random indices and check which files they come from
    n_samples = min(1000, len(dataset))
    indices = np.random.randint(0, len(dataset), n_samples)
    
    file_counts = Counter()
    for idx in indices:
        file_idx, _ = dataset._get_file_and_index(idx)
        file_counts[file_idx] += 1
    
    print(f"Sampled {n_samples} random indices across {len(dataset.file_paths)} files:")
    total_weight = sum(dataset.file_entries)
    for file_idx in sorted(file_counts.keys()):
        expected_ratio = dataset.file_entries[file_idx] / total_weight
        actual_ratio = file_counts[file_idx] / n_samples
        print(f"  File {file_idx}: {file_counts[file_idx]} samples "
              f"(expected: {expected_ratio:.3f}, actual: {actual_ratio:.3f})")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test ProngDatasetMultiFile")
    parser.add_argument("-f", "--filelist", type=str, required=True,
                        help="Path to file list with entry counts")
    parser.add_argument("-n", "--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("-b", "--batch-size", type=int, default=8,
                        help="Batch size for DataLoader test")
    parser.add_argument("--no-transform", action="store_true",
                        help="Don't apply normalization transform")
    
    args = parser.parse_args()
    
    # Create transforms
    if args.no_transform:
        transform = None
        print("Not using normalization transform")
    else:
        transform = transforms.Normalize(mean, std)
        print("Using normalization transform")
    
    # Create dataset
    print(f"Creating dataset from {args.filelist}...")
    try:
        dataset = ProngDatasetMultiFile(
            args.filelist,
            transformations=transform,
            clip=1000.0,
            cache_size=10,
            debug=False
        )
        print(f"Dataset created successfully with {len(dataset)} total entries")
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        return 1
    
    # Run tests
    tests_passed = 0
    tests_total = 1
    
    #if test_single_item_access(dataset):
    #    tests_passed += 1
    
    #if test_file_distribution(dataset):
    #    tests_passed += 1
    
    if test_dataloader(dataset, args.num_workers, args.batch_size):
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("All tests PASSED! ✓")
        return 0
    else:
        print("Some tests FAILED! ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
