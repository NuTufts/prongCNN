#!/usr/bin/env python3
"""
Script to create a file list with entry counts for ProngDatasetMultiFile.
Reads a list of ROOT files and outputs a formatted file with:
    <num_entries> <file_path>
"""

import argparse
import uproot
import os
from tqdm import tqdm

def count_entries_in_file(filepath):
    """Count number of entries in ImageTree of a ROOT file."""
    try:
        tree = uproot.open(filepath)["ImageTree"]
        return tree.num_entries
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Create file list with entry counts for ProngDatasetMultiFile")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input text file containing list of ROOT files (one per line)")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output file with format: <num_entries> <file_path>")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip files that don't exist or can't be read")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    
    args = parser.parse_args()
    
    # Read input file list
    with open(args.input, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Processing {len(file_paths)} files...")
    
    # Process files and collect results
    results = []
    total_entries = 0
    skipped = 0
    
    # Use tqdm for progress bar
    for filepath in tqdm(file_paths, desc="Counting entries"):
        if not os.path.exists(filepath):
            if args.verbose:
                print(f"File not found: {filepath}")
            if args.skip_missing:
                skipped += 1
                continue
            else:
                raise FileNotFoundError(f"File not found: {filepath}")
        
        num_entries = count_entries_in_file(filepath)
        
        if num_entries is None:
            if args.skip_missing:
                skipped += 1
                continue
            else:
                raise RuntimeError(f"Failed to read entries from {filepath}")
        
        results.append((num_entries, filepath))
        total_entries += num_entries
    
    # Write output file
    with open(args.output, 'w') as f:
        f.write("# File list for ProngDatasetMultiFile\n")
        f.write(f"# Total files: {len(results)}\n")
        f.write(f"# Total entries: {total_entries}\n")
        if skipped > 0:
            f.write(f"# Skipped files: {skipped}\n")
        f.write("# Format: <num_entries> <file_path>\n")
        f.write("#\n")
        
        for num_entries, filepath in results:
            f.write(f"{num_entries} {filepath}\n")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Processed files: {len(results)}")
    print(f"  Total entries: {total_entries}")
    if skipped > 0:
        print(f"  Skipped files: {skipped}")
    print(f"  Output written to: {args.output}")
    
    # Print distribution statistics
    if results:
        entries_list = [r[0] for r in results]
        print(f"\nEntry distribution:")
        print(f"  Min entries per file: {min(entries_list)}")
        print(f"  Max entries per file: {max(entries_list)}")
        print(f"  Avg entries per file: {sum(entries_list) / len(entries_list):.1f}")

if __name__ == "__main__":
    main()