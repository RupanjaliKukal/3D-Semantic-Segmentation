import torch 
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from utils import read_tiff, show_tiff

class ElectronMicroscopyDataset(torch.utils.data.Dataset):
    def __init__(self, img_tif, gt_tif, crop_size, slices, batch_size, iters=3200):
        super().__init__()
        self.img_tif = img_tif
        self.gt_tif = gt_tif
        self.slices = slices #depth
        self.crop_size = crop_size #for height and width (H x W)

        self.iters = iters 

        im_arr = read_tiff(self.img_tif) # (165, 768, 1024)
        self.im_arr = np.transpose(im_arr, (1,2,0)) # (768, 1024, 165)
        

        gt_arr = read_tiff(self.gt_tif) # (165, 768, 1024)
        self.gt_arr = np.transpose(gt_arr, (1,2,0)) # (768, 1024, 165)

        #dividing the dataset into train set
        self.im_arr = self.im_arr[...,:133] # (768, 1024, 133)
        self.gt_arr = self.gt_arr[...,:133] # (768, 1024, 133)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.iters

    def __getitem__(self, idx):
        
        x = np.random.randint(0, self.im_arr.shape[0] - self.crop_size[0])
        y = np.random.randint(0, self.im_arr.shape[1] - self.crop_size[1])
        z = np.random.randint(0, self.im_arr.shape[2] - self.slices)

        voxel = self.im_arr[x:x+self.crop_size[0] , y:y+self.crop_size[1], z:z+self.slices]
        voxel = self.transform(voxel)
        voxel = np.transpose(voxel, (1,2,0))

        gt = self.gt_arr[x:x+self.crop_size[0] , y:y+self.crop_size[1], z:z+self.slices]
        gt = self.transform(gt)
        gt = np.transpose(gt, (1,2,0))

        sample = {'voxel' : voxel, 'gt' : gt }

        return sample


class ElectronMicroscopyDataset_Test(torch.utils.data.Dataset):
    def __init__(self, img_tif, gt_tif, crop_size, slices, batch_size, mode = 'val', iters=3200):
        super().__init__()
        self.img_tif = img_tif
        self.gt_tif = gt_tif
        self.slices = slices #depth
        self.crop_size = crop_size #for height and width (H x W)

        self.iters = iters 

        im_arr = read_tiff(self.img_tif) # (165, 768, 1024)
        self.im_arr = np.transpose(im_arr, (1,2,0)) # (768, 1024, 165)
        
        gt_arr = read_tiff(self.gt_tif) # (165, 768, 1024)
        self.gt_arr = np.transpose(gt_arr, (1,2,0)) # (768, 1024, 165)

        #dividing the dataset into validation data
        if mode == 'val':
            self.im_arr = self.im_arr[...,133:] # (768, 1024, 32)
            self.gt_arr = self.gt_arr[...,133:] # (768, 1024, 32)


        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.im_chunk = []
        self.gt_chunk = []

        for x in range(self.im_arr.shape[0]//self.crop_size[0]):
            for y in range(self.im_arr.shape[1]//self.crop_size[1]):
                for z in range(self.im_arr.shape[2]//self.slices):

                    self.im_chunk.append(self.im_arr[x*self.crop_size[0]:x*self.crop_size[0]+self.crop_size[0] , y*self.crop_size[1]:y*self.crop_size[1]+self.crop_size[1], z*self.slices:z*self.slices+self.slices])
                    self.gt_chunk.append(self.gt_arr[x*self.crop_size[0]:x*self.crop_size[0]+self.crop_size[0] , y*self.crop_size[1]:y*self.crop_size[1]+self.crop_size[1], z*self.slices:z*self.slices+self.slices])
        
        self.im_chunk = np.array(self.im_chunk)
        self.gt_chunk = np.array(self.gt_chunk)


    def __len__(self):
        return self.im_chunk.shape[0]

    def __getitem__(self, idx):

        voxel = self.im_chunk[idx]
        voxel = self.transform(voxel)
        voxel = np.transpose(voxel, (1,2,0))

        gt = self.gt_chunk[idx]
        gt = self.transform(gt)
        gt = np.transpose(gt, (1,2,0))
        
        sample = {'voxel' : voxel, 'gt' : gt }

        return sample


def main():
    root = "./data"
    
    ##to visualize slices 
    # view_tiff = show_tiff(f"{root}/training.tif", f"{root}/visualize")

    img_tif = f"{root}/training.tif"
    gt_tif = f"{root}/training_groundtruth.tif"
    crop_size = [300,300]
    slices = 30
    batch_size = 64

    train_dataset = ElectronMicroscopyDataset(img_tif, gt_tif, crop_size, slices, batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    # sample1 = next(iter(train_dataset))
    # sample2 = next(iter(train_dataloader))
    # visualize(sample2['voxel'])


if __name__ == '__main__':
    main()