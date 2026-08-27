import torch
import torchvision
import os 
from tqdm import tqdm
from torchvision.utils import make_grid
import math
from PIL import Image

#We will first define the diffusion tools
#1. Setting up Noise Schedule
#2. Function for noising images
#3. function for sampling images

class Diffusion:
    def __init__(self, device, noise_steps = 2000, beta_start = 1e-6, beta_end = 1e-2, img_size = 256):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        #Precomputing all alpha and beta values
        self.beta = torch.linspace(start = self.beta_start, end = self.beta_end, steps = self.noise_steps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)
    
    def return_alpha_hat(self):
        alpha_hat =  self.alpha_hat
        return alpha_hat
        
    #function to generate noisy images at timestep t    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        
        return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*eps, eps
    
    #function to sample a random timestep
    def sample_timesteps(self, n):
        
        return torch.randint(low = 0, high = self.noise_steps, size = (n,))
    
    # def load_conditions(self, test_dir, num_samples, device, transform):

    #     img_files = sorted(os.listdir(test_dir))

    #     cond_imgs = []
    #     for i in range(num_samples):
    #         img = Image.open(os.path.join(test_dir, img_files[i])).convert("RGB")
    #         img = transform(img)
            
    #         cond_imgs.append(img)   #no need to concat depth if not used 

    #     condition = torch.stack(cond_imgs, dim=0).to(device)
    #     return condition
    
    #function which will take in the model and the number of images we want to sample from the model    
    def sample(self, model, test_dir = None, transform = None): 
        #Following Algorithm 2 from DDPM paper
        model.eval()

        #Defining the testset
        with torch.no_grad():
            for imgname in os.listdir(test_dir):
                img = Image.open(os.path.join(os.path.join(test_dir, imgname))).convert("RGB")
                img = transform(img)
                cond = img.unsqueeze(0)
                cond = cond.to(self.device)
                x = torch.randn((1, 3, self.img_size, self.img_size)).to(self.device)
                # print("x shape:", x.shape, "cond shape:", cond.shape)
                for i in tqdm(reversed(range(0, self.noise_steps)), position = 0):
                    t = (torch.ones(1)*i).long().to(self.device)
                    # t = torch.full((n,), i, dtype=torch.long, device=self.device)
                    '''model will predict the residual by reverse sampling through noise steps'''
                    predicted_noise = model(torch.concat([x, cond], dim = 1), t)
                    
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    
                    #for the last step we want the noise to be zero
                    if i>1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    
                    x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1 - alpha_hat)))* predicted_noise) + torch.sqrt(beta)*noise 
                    
                    if i==0:
                        '''conversion of residual to full image'''
                        reconstructed = cond + x
                        imsx = torch.clamp(x, -1, 1).detach().cpu()
                        imsx = (imsx+1)/2
                        #Split reconstructed residual channels
                        R = imsx[:, 0:1, :, :]
                        G = imsx[:, 1:2, :, :]
                        B = imsx[:, 2:3, :, :]


                        ims = torch.clamp(reconstructed, -1, 1).detach().cpu()
                        ims = (ims + 1)/2

                        ims = ims.squeeze(0)
                        R = R.squeeze(0)
                        G = G.squeeze(0)
                        B = B.squeeze(0)

                        img = torchvision.transforms.ToPILImage()(ims)
                        xR = torchvision.transforms.ToPILImage()(R)
                        xG = torchvision.transforms.ToPILImage()(G)
                        xB = torchvision.transforms.ToPILImage()(B)

                        
                        os.makedirs('./Generated_Samples', exist_ok = True)
                        os.makedirs('./Generated_Samples/Images', exist_ok = True)
                        os.makedirs('./Generated_Samples/R', exist_ok = True)
                        os.makedirs('./Generated_Samples/G', exist_ok = True)
                        os.makedirs('./Generated_Samples/B', exist_ok = True)
                        img.save(os.path.join('./Generated_Samples/Images', imgname))
                        xR.save(os.path.join('./Generated_Samples/R', imgname))
                        xG.save(os.path.join('./Generated_Samples/G', imgname))
                        xB.save(os.path.join('./Generated_Samples/B', imgname))
                        img.close()
                        xR.close()
                        xB.close()
                        xG.close()
                        print(f"Image and RGB residuals saved for {imgname}")
            
                
        # model.train()
        # x = (x.clamp(-1, 1)+1)/2
        # x = (x*255).type(torch.uint8)
        # return x    