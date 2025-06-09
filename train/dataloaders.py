import os,sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/models')
from datasets_reco_5ClassHardLabel_quadTask import ProngDataset, mean, std

def make_dataloaders( train_file, val_file,
                      batch_size_train, batch_size_val,
                      num_workers,
                      img_mean=mean, img_std=std,
                      noMask=False, multiTask=False,
                      classifyComp=False, dropLast=True ):
    
    print("CREATING DATA LOADERS")
    if noMask:
      img_mean = mean_nm
      img_std = std_nm
    train_transform = transforms.Compose([transforms.Normalize(img_mean, img_std),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomVerticalFlip(0.5)])
    test_transform = transforms.Normalize(img_mean, img_std)

    if noMask:
      train_dataset = ProngDatasetNoMask(train_file, transformations=train_transform, clip=4.0)
      test_dataset = ProngDatasetNoMask(val_file, transformations=test_transform, clip=4.0)
    else:
      if multiTask and classifyComp:
        train_dataset = ProngDatasetClCmp(train_file, transformations=train_transform, clip=4.0)
        test_dataset = ProngDatasetClCmp(val_file, transformations=test_transform, clip=4.0)
      else:
        train_dataset = ProngDataset(train_file, transformations=train_transform, clip=4.0)
        test_dataset = ProngDataset(val_file, transformations=test_transform, clip=4.0)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, drop_last=dropLast, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_val, drop_last=False, shuffle=True, num_workers=num_workers)

    dataloaders = {
        'train_dataset':train_dataset,
        'val_dataset':test_dataset,
        'train_dataloader':train_dataloader,
        'val_dataloader':test_dataloader
    }
    
    return dataloaders


if __name__ == "__main__":
    print("Test data loader")
    train_file = "../prongcnndata_filtered_badfilesremoved_shuffled_2000PerClassVal_train.root"
    val_file   = "../prongcnndata_filtered_badfilesremoved_shuffled_2000PerClassVal_test.root"
    batchsize_train = 8
    batchsize_valid = 8
    num_workers = 0
    num_iters = 1000
    
    dataloaders = make_dataloaders( train_file, val_file, batchsize_train, batchsize_valid, num_workers )
    print("made data loaders")

    iiter = 0
    for batch, (X,y) in enumerate(dataloaders['train_dataloader']):
        print("batch: ",batch, (X,y))
        iiter += 1
        if iiter >=num_iters:
            break

    print("finished test loop")
        
