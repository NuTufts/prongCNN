
import uproot
import numpy as np
import torch
from torch.utils.data import Dataset

mean = (57.8182, 57.8182, 58.1807, 58.1807, 50.5312, 50.5312)
std = (62.9932, 62.9932, 62.6569, 62.6569, 42.0027, 42.0027)

#meanPl2 = (50.5312, 50.5312)
#stdPl2 = (42.0027, 42.0027)
#
#mean_nm = (57.8182, 58.1807, 50.5312)
#std_nm = (62.9932, 62.6569, 42.0027)
#
#meanPl2_nm = 50.5312
#stdPl2_nm = 42.0027

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



class ProngDataset(Dataset):
    
    def __init__(self, rootfile, transformations=None, clip=1000.0):
        self.filename = rootfile
        tree = uproot.open(self.filename)["ImageTree"]
        #pdgs = tree["pdg"].array(library="np")
        #self.classes = np.array([getClass(pdgs[i]) for i in range(len(pdgs))])
        arrays = tree.arrays(["pdg", "processClass"], library="np")
        self.classes = np.array([getClass(arrays["pdg"][i]) for i in range(len(arrays["pdg"]))])
        self.procClasses = np.array([arrays["processClass"][i] for i in range(len(arrays["processClass"]))])
        self.transforms = transformations
        self.clipVal = clip
        self.keys = [
          "pdg",
          "completeness",
          "purity",
          "processClass",
          "plane0pix_row", "plane0pix_col", "plane0pix_val",
          "plane1pix_row", "plane1pix_col", "plane1pix_val",
          "plane2pix_row", "plane2pix_col", "plane2pix_val",
          "raw_plane0pix_row", "raw_plane0pix_col", "raw_plane0pix_val",
          "raw_plane1pix_row", "raw_plane1pix_col", "raw_plane1pix_val",
          "raw_plane2pix_row", "raw_plane2pix_col", "raw_plane2pix_val"]
    
    def __getitem__(self, item):
        print("retrieving ProngDataset entry", item)
        tree = uproot.open(self.filename)["ImageTree"]
        image = np.zeros((6,512,512))
        
        arrays = tree.arrays(["pdg", "completeness", "purity", "processClass",
                              "plane0pix_row", "plane0pix_col", "plane0pix_val",
                              "plane1pix_row", "plane1pix_col", "plane1pix_val",
                              "plane2pix_row", "plane2pix_col", "plane2pix_val",
                              "raw_plane0pix_row", "raw_plane0pix_col", "raw_plane0pix_val",
                              "raw_plane1pix_row", "raw_plane1pix_col", "raw_plane1pix_val",
                              "raw_plane2pix_row", "raw_plane2pix_col", "raw_plane2pix_val"],
                             library="np", entry_start=item, entry_stop=item+1)
        # for debug
        for k in self.keys:
          print(k,": ",arrays[k].shape)
        
        image[0, arrays["plane0pix_row"][0], arrays["plane0pix_col"][0]] = arrays["plane0pix_val"][0]
        image[2, arrays["plane1pix_row"][0], arrays["plane1pix_col"][0]] = arrays["plane1pix_val"][0]
        image[4, arrays["plane2pix_row"][0], arrays["plane2pix_col"][0]] = arrays["plane2pix_val"][0]
        image[1, arrays["raw_plane0pix_row"][0], arrays["raw_plane0pix_col"][0]] = arrays["raw_plane0pix_val"][0]
        image[3, arrays["raw_plane1pix_row"][0], arrays["raw_plane1pix_col"][0]] = arrays["raw_plane1pix_val"][0]
        image[5, arrays["raw_plane2pix_row"][0], arrays["raw_plane2pix_col"][0]] = arrays["raw_plane2pix_val"][0]
        
        image = torch.from_numpy(image).float()
        target = [ getClass(arrays["pdg"][0]), arrays["completeness"][0], arrays["purity"][0], arrays["processClass"][0] ]
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return torch.clamp(image, max=self.clipVal), target
        
    def __len__(self):
        tree = uproot.open(self.filename)["ImageTree"]
        return tree.num_entries



#other versions below not modified for multiTask
    
#class ProngDatasetPl2(Dataset):
#    
#    def __init__(self, rootfile, transformations=None, clip=1000.0):
#        self.filename = rootfile
#        tree = uproot.open(self.filename)["ImageTree"]
#        pdgs = tree["pdg"].array(library="np")
#        self.classes = np.array([getClass(pdgs[i]) for i in range(len(pdgs))])
#        self.transforms = transformations
#        self.clipVal = clip
#    
#    def __getitem__(self, item):
#        #print("retrieving ProngDataset entry", item)
#        tree = uproot.open(self.filename)["ImageTree"]
#        image = np.zeros((2,512,512))
#        
#        arrays = tree.arrays(["pdg", "plane2pix_row", "plane2pix_col", "plane2pix_val",
#                              "raw_plane2pix_row", "raw_plane2pix_col", "raw_plane2pix_val"],
#                             library="np", entry_start=item, entry_stop=item+1)
#        image[0, arrays["plane2pix_row"][0], arrays["plane2pix_col"][0]] = arrays["plane2pix_val"][0]
#        image[1, arrays["raw_plane2pix_row"][0], arrays["raw_plane2pix_col"][0]] = arrays["raw_plane2pix_val"][0]
#        
#        image = torch.from_numpy(image).float()
#        Class = getClass(arrays["pdg"][0])
#        
#        if self.transforms is not None:
#            image = self.transforms(image)
#        
#        return torch.clamp(image, max=self.clipVal), Class
#        
#    def __len__(self):
#        tree = uproot.open(self.filename)["ImageTree"]
#        return tree.num_entries
#
#
#class ProngDatasetNoMask(Dataset):
#    
#    def __init__(self, rootfile, transformations=None, clip=1000.0):
#        self.filename = rootfile
#        tree = uproot.open(self.filename)["ImageTree"]
#        pdgs = tree["pdg"].array(library="np")
#        self.classes = np.array([getClass(pdgs[i]) for i in range(len(pdgs))])
#        self.transforms = transformations
#        self.clipVal = clip
#    
#    def __getitem__(self, item):
#        #print("retrieving ProngDataset entry", item)
#        tree = uproot.open(self.filename)["ImageTree"]
#        image = np.zeros((3,512,512))
#        
#        arrays = tree.arrays(["pdg", "plane0pix_row", "plane0pix_col", "plane0pix_val",
#                              "plane1pix_row", "plane1pix_col", "plane1pix_val",
#                              "plane2pix_row", "plane2pix_col", "plane2pix_val"],
#                             library="np", entry_start=item, entry_stop=item+1)
#        image[0, arrays["plane0pix_row"][0], arrays["plane0pix_col"][0]] = arrays["plane0pix_val"][0]
#        image[1, arrays["plane1pix_row"][0], arrays["plane1pix_col"][0]] = arrays["plane1pix_val"][0]
#        image[2, arrays["plane2pix_row"][0], arrays["plane2pix_col"][0]] = arrays["plane2pix_val"][0]
#        
#        image = torch.from_numpy(image).float()
#        Class = getClass(arrays["pdg"][0])
#        
#        if self.transforms is not None:
#            image = self.transforms(image)
#        
#        return torch.clamp(image, max=self.clipVal), Class
#        
#    def __len__(self):
#        tree = uproot.open(self.filename)["ImageTree"]
#        return tree.num_entries
#
#    
#class ProngDatasetPl2NoMask(Dataset):
#    
#    def __init__(self, rootfile, transformations=None, clip=1000.0):
#        self.filename = rootfile
#        tree = uproot.open(self.filename)["ImageTree"]
#        pdgs = tree["pdg"].array(library="np")
#        self.classes = np.array([getClass(pdgs[i]) for i in range(len(pdgs))])
#        self.transforms = transformations
#        self.clipVal = clip
#    
#    def __getitem__(self, item):
#        #print("retrieving ProngDataset entry", item)
#        tree = uproot.open(self.filename)["ImageTree"]
#        image = np.zeros((1,512,512))
#        
#        arrays = tree.arrays(["pdg", "plane2pix_row", "plane2pix_col", "plane2pix_val"],
#                             library="np", entry_start=item, entry_stop=item+1)
#        image[0, arrays["plane2pix_row"][0], arrays["plane2pix_col"][0]] = arrays["plane2pix_val"][0]
#        
#        image = torch.from_numpy(image).float()
#        Class = getClass(arrays["pdg"][0])
#        
#        if self.transforms is not None:
#            image = self.transforms(image)
#        
#        return torch.clamp(image, max=self.clipVal), Class
#        
#    def __len__(self):
#        tree = uproot.open(self.filename)["ImageTree"]
#        return tree.num_entries


