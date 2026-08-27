import torch
from torch.utils.data import  Dataset
import os
from PIL import Image

#with depth condition
class Wide2LongDatasetAfterInitialEstimate(Dataset):
    def __init__(self, dir_concatdataset, dir_estimate, transform_concatdataset = None, transform_estimate = None, transform_aug = None):
        self.dir_concatdataset = dir_concatdataset
        self.dir_estimate = dir_estimate
        self.transform_concatdataset = transform_concatdataset
        self.transform_estimate = transform_estimate
        self.paths_concatdataset = []
        self.paths_estimate = []
        self.transform_aug = transform_aug
        
        for gtname in os.listdir(self.dir_concatdataset):    #Lets first see through all the images with inputs and GT concatenated
            x = ''
            #Extract x from imgx.jpg or imgx.png
            if gtname.endswith(".jpg"):
                x = gtname.replace("img", "").replace(".jpg", "")
            elif gtname.endswith(".png"): 
                x = gtname.replace("img", "").replace(".png", "")
            if(int(x)<=500):
                #We'll see if that exact name exists in the dir_estimate folder
                estname = "output" + x + '.png'
                if(os.path.exists(os.path.join(self.dir_estimate, estname))):
                    self.paths_concatdataset.append(os.path.join(self.dir_concatdataset, gtname))
                    self.paths_estimate.append(os.path.join(self.dir_estimate, estname))
                    # print(gtname)
                else:
                    print(f"Warning: No initial estimate found for {gtname}")
                    
        
    def __len__(self):
        return len(self.paths_concatdataset)
    
    def __getitem__(self, index):
        img_concatdataset = Image.open(self.paths_concatdataset[index]).convert("RGB")
        img_estimate = Image.open(self.paths_estimate[index]).convert("RGB")
        
        img_GT = self.transform_concatdataset(img_concatdataset)
        # print(img_GT.shape)
        img_GT = img_GT[:, :, 512:]
        img_estimate = self.transform_estimate(img_estimate)

        if self.transform_aug is not None:
            combined = torch.cat([img_estimate, img_GT], dim = 0)
            combined = self.transform_aug(combined)
            img_estimate = combined[:3]
            img_GT = combined[3:]
        
        return img_estimate, img_GT
    
