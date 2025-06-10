import uproot
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import bisect
from threading import Lock
import traceback

# Use same normalization constants as original
mean = (57.8182, 57.8182, 58.1807, 58.1807, 50.5312, 50.5312)
std  = (62.9932, 62.9932, 62.6569, 62.6569, 42.0027, 42.0027)

def getClass(pid):
    pid = abs(pid)
    if pid == 11:
        return 0 
    if pid == 22:
        return 1 
    if pid == 13:
        return 2 
    if pid == 211:
        return 3
    if pid == 2212:
        return 4
    return -1


class ProngDatasetMultiFile(Dataset):
    """
    PyTorch Dataset that reads from multiple ROOT files listed in a text file.
    The text file should have two columns:
    - Column 1: Number of entries in the file
    - Column 2: Path to the ROOT file
    
    This dataset implements weighted sampling where files are sampled proportionally
    to their number of entries, ensuring uniform sampling across all examples.
    """
    
    def __init__(self, filelist_path, transformations=None, clip=1000.0, cache_size=10, debug=False):
        """
        Args:
            filelist_path: Path to text file with format: <num_entries> <file_path>
            transformations: Optional torchvision transforms to apply
            clip: Maximum value to clip pixel values
            cache_size: Number of open file handles to cache (for multiprocessing efficiency)
            debug: if True (default: False), will print out info to understand data
        """
        self.transforms = transformations
        self.clipVal = clip
        self.cache_size = cache_size
        self.debug = debug
        self.filelist_path = filelist_path
        self.total_entries = 0

        # Keys to extract from ROOT files
        self.keys = [
            "pdg", "completeness", "purity", "processClass",
            "plane0pix_row", "plane0pix_col", "plane0pix_val",
            "plane1pix_row", "plane1pix_col", "plane1pix_val",
            "plane2pix_row", "plane2pix_col", "plane2pix_val",
            "raw_plane0pix_row", "raw_plane0pix_col", "raw_plane0pix_val",
            "raw_plane1pix_row", "raw_plane1pix_col", "raw_plane1pix_val",
            "raw_plane2pix_row", "raw_plane2pix_col", "raw_plane2pix_val"
        ]

    def load_filelists(self):
        
        # Parse the file list
        self.file_paths = []
        self.file_entries = []
        self.cumulative_entries = [0]
        total_entries = 0
        
        with open(self.filelist_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: {line}")
                
                num_entries = int(parts[0])
                file_path = parts[1]
                
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"ROOT file not found: {file_path}")
                
                self.file_paths.append(file_path)
                self.file_entries.append(num_entries)
                total_entries += num_entries
                self.cumulative_entries.append(total_entries)
        
        self.total_entries = total_entries
        
        # File handle cache for multiprocessing efficiency
        # Each worker will maintain its own cache
        #self._file_cache = {}
        #self._cache_order = []
        #self._cache_lock = Lock()
        
        
        print(f"Initialized ProngDatasetMultiFile with {len(self.file_paths)} files, {self.total_entries} total entries")
    
    def _get_file_and_index(self, global_index):
        """
        Convert global index to (file_index, local_index) pair.
        Uses binary search for efficiency.
        """
        if global_index >= self.total_entries:
            raise IndexError(f"Index {global_index} out of range [0, {self.total_entries})")
        
        # Binary search to find which file contains this index
        file_idx = bisect.bisect_right(self.cumulative_entries, global_index)
        
        # Calculate local index within the file
        if file_idx == 0:
            local_idx = global_index
        else:
            local_idx = global_index - self.cumulative_entries[file_idx - 1]
        
        return file_idx, local_idx
    
    def __getitem__(self, item):
        """
        Get a single example by global index.
        """

        if not hasattr(self,'file_entries'):
            self.load_filelists()

        file_idx = np.searchsorted(self.cumulative_entries, item, side='right') - 1
        local_idx = item - self.cumulative_entries[file_idx]
        
        # Convert global index to file and local index
        #file_idx, local_idx = self._get_file_and_index(item)
        file_path = self.file_paths[file_idx]
        
        # Debug print (can be removed in production)
        if self.debug and item % 1000 == 0:
            print(f"ProngDatasetMultiFile: retrieving entry {item} (file {file_idx}, local {local_idx})")
        
        # Get tree (from cache if available)
        #tree = self._get_tree(file_path)
        tree = uproot.open(file_path)["ImageTree"]        
        
        # Initialize empty image
        image = np.zeros((6, 512, 512))
        
        # Read arrays for this specific entry
        arrays = tree.arrays(self.keys, library="np", 
                             entry_start=local_idx, entry_stop=local_idx+1)
        
        # Fill image channels
        # Channels 0, 2, 4: prong pixels for planes 0, 1, 2
        # Channels 1, 3, 5: context (raw) pixels for planes 0, 1, 2
        if self.debug:
            for k in self.keys:
                print(k,": ",arrays[k][0].shape)

        try:
            image[0, arrays["plane0pix_row"][0], arrays["plane0pix_col"][0]] = arrays["plane0pix_val"][0]
            image[2, arrays["plane1pix_row"][0], arrays["plane1pix_col"][0]] = arrays["plane1pix_val"][0]
            image[4, arrays["plane2pix_row"][0], arrays["plane2pix_col"][0]] = arrays["plane2pix_val"][0]
            image[1, arrays["raw_plane0pix_row"][0], arrays["raw_plane0pix_col"][0]] = arrays["raw_plane0pix_val"][0]
            image[3, arrays["raw_plane1pix_row"][0], arrays["raw_plane1pix_col"][0]] = arrays["raw_plane1pix_val"][0]
            image[5, arrays["raw_plane2pix_row"][0], arrays["raw_plane2pix_col"][0]] = arrays["raw_plane2pix_val"][0]
        except Exception:
            print("Error loading the image arrays")
            print(traceback.format_exc())
            print("file loaded: ",file_path)
            print("file index, local index: ",(file_idx,local_idx))
            raise ValueError("Error loading the image arrays")
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Prepare target (list format to match original dataset)
        target = [
            getClass(arrays["pdg"][0]), 
            arrays["completeness"][0], 
            arrays["purity"][0], 
            arrays["processClass"][0]
        ]
        
        # Apply transforms if any
        if self.transforms is not None:
            image = self.transforms(image)
        
        # Clip values and return
        return torch.clamp(image, max=self.clipVal), target
    
    def __len__(self):
        """Return total number of entries across all files."""
        if not hasattr(self,'file_entries'):
            self.load_filelists()
        
        return self.total_entries
    
    def get_file_weights(self):
        """
        Return weights for each file proportional to number of entries.
        Useful for debugging or analysis.
        """
        return np.array(self.file_entries) / self.total_entries
    
    def create_filelist(self, root_files, output_path):
        """
        Helper method to create a properly formatted file list.
        
        Args:
            root_files: List of ROOT file paths
            output_path: Where to save the file list
        """
        with open(output_path, 'w') as f:
            f.write("# num_entries file_path\n")
            for file_path in root_files:
                try:
                    tree = uproot.open(file_path)["ImageTree"]
                    num_entries = tree.num_entries
                    f.write(f"{num_entries} {file_path}\n")
                    print(f"Added {file_path}: {num_entries} entries")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


# For backward compatibility, export the class under the original name too
ProngDataset = ProngDatasetMultiFile
