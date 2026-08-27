#%%
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def return_patchwiseSSIM(img1, img2, patch_size = 9, padding = 4):
    #Preparing the images with padding
    img1_padded = np.zeros((img1.shape[0] + 2*padding, img1.shape[1] + 2*padding, 3), dtype = np.float32)
    img2_padded = np.zeros((img2.shape[0] + 2*padding, img2.shape[1] + 2*padding, 3), dtype = np.float32)
    img1_padded[padding:img1.shape[0]+padding,padding:img1.shape[1]+padding, :] = img1
    img2_padded[padding:img2.shape[0]+padding,padding:img2.shape[1]+padding, :] = img2

    #Declare the SSIM map
    SSIM_map = np.zeros((img1.shape[0], img1.shape[1]), dtype = np.float32)

    for i in range(padding, padding+img1.shape[0]):
        for j in range(padding, padding+img2.shape[0]):
            patch1 = img1_padded[i-patch_size//2:i+patch_size//2+1, j-patch_size//2:j+patch_size//2+1, :]
            patch2 = img2_padded[i-patch_size//2:i+patch_size//2+1, j-patch_size//2:j+patch_size//2+1, :]
            patch_score = ssim(patch1, patch2, data_range = 255, channel_axis = -1)     #[-1, 1]
            SSIM_map[i-padding, j-padding] = patch_score
    
    # return np.mean(SSIM_map)
    return SSIM_map

GTdir = 'Dataset/f2fGT'
# DiffOutputdir = 'Generated_Samples/Generated_Samples_LinearTrainSigmoidSample(a=8)L10.01FidelityLoss/Images'
DiffOutputdir = 'Generated_Samples/Bilateral_Filtered_29_40_20/Images'
W2LOutputdir = 'Dataset/testestimatesW2L'

#%%
for imgname in os.listdir(W2LOutputdir):
    DiffOutputfoldername = imgname.replace(".png", "")
    W2Limgdir = os.path.join(W2LOutputdir, imgname)
    Diffimgdir = os.path.join(DiffOutputdir, DiffOutputfoldername)
    Diffimgdir = os.path.join(Diffimgdir, '0'+DiffOutputfoldername+'.png')
    imgnum = DiffOutputfoldername.replace("output", "")
    GTimgdir = os.path.join(GTdir, "img"+imgnum+".jpg")

    imgDiff = cv2.imread(Diffimgdir)
    imgW2L = cv2.imread(W2Limgdir)
    imgGT = cv2.imread(GTimgdir)
    imgGT = imgGT[:, 512:, :]

    DiffSSIM = return_patchwiseSSIM(img1 = imgDiff, img2 = imgGT, patch_size = 9, padding = 4)
    W2LSSIM = return_patchwiseSSIM(img1 = imgW2L, img2 = imgGT, patch_size = 9, padding = 4)

    if W2LSSIM>DiffSSIM:    #point out the worse Diffusion outputs
        print(f"For {imgname}: W2LvsGTSSIM: {W2LSSIM}, DiffvsGTSSIM: {DiffSSIM}")

    
# %%
#The image numbers with worse SSIMs for Diffusion chosen are: 510, 511, 513, 521, 522
imgnums = ['510', '511', '513', '521', '522']

for i in imgnums:
    imgname = 'output'+i+'.png'
    W2Limgdir = os.path.join(W2LOutputdir, imgname)
    Diffimgdir = os.path.join(DiffOutputdir, imgname.replace(".png", ""))
    Diffimgdir = os.path.join(Diffimgdir, '0output'+i+'.png')
    GTimgdir = os.path.join(GTdir, "img"+i+".jpg")

    imgDiff = cv2.imread(Diffimgdir)
    imgW2L = cv2.imread(W2Limgdir)
    imgGT = cv2.imread(GTimgdir)
    imgGT = imgGT[:, 512:, :]

    SSIMMapDiff = return_patchwiseSSIM(img1 = imgDiff, img2 = imgGT, patch_size = 9, padding = 4)
    SSIMMapW2L = return_patchwiseSSIM(img1 = imgW2L, img2 = imgGT, patch_size = 9, padding = 4)

    #Diffusion - W2L
    DifferenceMap = SSIMMapDiff - SSIMMapW2L    #[-2, 2]

    DifferenceMap_Thresholded = np.where(DifferenceMap>=0, 1, 0)

    # plt.imshow(DifferenceMap_Thresholded, cmap = 'gray', vmax = 1, vmin = 0)
    # plt.show()
    DifferenceMapRGB = np.zeros_like(imgDiff, dtype = np.uint8)
    
    #Green for 1
    DifferenceMapRGB[DifferenceMap_Thresholded==1] = [0, 255, 0]
    #Red for 0
    DifferenceMapRGB[DifferenceMap_Thresholded==0] = [0, 0, 255]    #BGR

    cv2.imwrite("DiffGreenRed" + i + '.png', img = DifferenceMapRGB)
    print("Image " + i + " saved.")


# %%
