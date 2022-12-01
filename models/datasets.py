
import uproot
import numpy as np
import torch
from torch.utils.data import Dataset

mean = (57.8182, 58.1807, 50.5312)
std = (62.9932, 62.6569, 42.0027)

meanPl2 = 50.5312
stdPl2 = 42.0027


def getClass(pid):
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


def getClass3part(pid):
  if pid == 11:
    return 0 
  if pid == 211:
    return 1
  if pid == 2212:
    return 2
  return -1


class ProngDataset(Dataset):
    
    def __init__(self, rootfile, transformations=None, clip=1000.0, threePart=False):
        self.file = uproot.open(rootfile)
        self.tree = self.file["ImageTree"]
        pdgs = self.tree["pdg"].array(library="np")
        self.classes = np.array([getClass(pdgs[i]) for i in range(len(pdgs))])
        self.transforms = transformations
        self.threePartClass = threePart
        self.clipVal = clip
    
    def __getitem__(self, item):
        #print("retrieving ProngDataset entry", item)
        image = np.zeros((3,512,512))
        
        arrays = self.tree.arrays(["pdg", "plane0pix_row", "plane0pix_col", "plane0pix_val",
                                   "plane1pix_row", "plane1pix_col", "plane1pix_val",
                                   "plane2pix_row", "plane2pix_col", "plane2pix_val"],
                                  library="np", entry_start=item, entry_stop=item+1)
        image[0, arrays["plane0pix_row"][0], arrays["plane0pix_col"][0]] = arrays["plane0pix_val"][0]
        image[1, arrays["plane1pix_row"][0], arrays["plane1pix_col"][0]] = arrays["plane1pix_val"][0]
        image[2, arrays["plane2pix_row"][0], arrays["plane2pix_col"][0]] = arrays["plane2pix_val"][0]
        
        image = torch.from_numpy(image).float()
        if self.threePartClass:
            Class = getClass3part(arrays["pdg"][0])
        else:
            Class = getClass(arrays["pdg"][0])
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return torch.clamp(image, max=self.clipVal), Class
        
    def __len__(self):
        return self.tree.num_entries

    
class ProngDatasetPl2(Dataset):
    
    def __init__(self, rootfile, transformations=None, clip=1000.0, threePart=False):
        self.file = uproot.open(rootfile)
        self.tree = self.file["ImageTree"]
        pdgs = self.tree["pdg"].array(library="np")
        self.classes = np.array([getClass(pdgs[i]) for i in range(len(pdgs))])
        self.transforms = transformations
        self.threePartClass = threePart
        self.clipVal = clip
    
    def __getitem__(self, item):
        #print("retrieving ProngDataset entry", item)
        image = np.zeros((1,512,512))
        
        arrays = self.tree.arrays(["pdg", "plane2pix_row", "plane2pix_col", "plane2pix_val"],
                                  library="np", entry_start=item, entry_stop=item+1)
        image[0, arrays["plane2pix_row"][0], arrays["plane2pix_col"][0]] = arrays["plane2pix_val"][0]
        
        image = torch.from_numpy(image).float()
        if self.threePartClass:
            Class = getClass3part(arrays["pdg"][0])
        else:
            Class = getClass(arrays["pdg"][0])
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return torch.clamp(image, max=self.clipVal), Class
        
    def __len__(self):
        return self.tree.num_entries


