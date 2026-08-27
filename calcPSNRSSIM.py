#%%
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def returnPSNRSSIM(img1, img2):
    score = ssim(img1, img2, data_range = 255, channel_axis = -1)
    db = psnr(img1, img2, data_range = 255)

    return db, score

concat_dir = 'Dataset/f2fGT'
gen_sam_dir = 'Generated_Samples'

for cases in os.listdir(gen_sam_dir):
    SSIMs = []
    PSNRs = []
    for imgname in os.listdir(os.path.join(gen_sam_dir, os.path.join(cases, 'Images'))):
        img1name = '0' + imgname + '.png'
        imgnum = imgname.replace("output", "")
        img2name = 'img' + imgnum + ".jpg"
        if os.path.exists(os.path.join(concat_dir, img2name)):
            # print(os.path.join(os.path.join(gen_sam_dir, cases),os.path.join("Images", os.path.join(imgname, img1name))))
            img1 = cv2.imread(os.path.join(os.path.join(gen_sam_dir, cases),os.path.join("Images", os.path.join(imgname, img1name))))
            img2 = cv2.imread(os.path.join(concat_dir, img2name))
            img2 = img2[:, 512:, :]
            db, score = returnPSNRSSIM(img1, img2)
            # print(f"For image {imgnum}, PSNR: ", db, "SSIM: ", score)
            PSNRs.append(db)
            SSIMs.append(score)
        else:
            print(f"No GT image found corresponding to {img1name}")

    print('For ', cases, ': ')
    print('PSNR: ', np.mean(PSNRs), '  SSIM: ', np.mean(SSIMs))


  # %%
