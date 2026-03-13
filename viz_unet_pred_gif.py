import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio

from wrf_vit_dataset import WrfVitShardDataset


# -------- CONFIG --------

SHARDS_DIR = "/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards"
CKPT_PATH  = "/home/abazan/wrfout_sandbox/unet_ckpt.pt"

FIRE_THR = 0.1
PB_THR   = float(os.environ.get("PB_THR", "0.8"))

OUT_GIF = "/home/abazan/wrfout_sandbox/unet_fire_prediction.gif"

FPS = 2


# -------- MODEL --------

class Block(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,3,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.net(x)


class UNet(nn.Module):

    def __init__(self,in_c,out_c):

        super().__init__()

        self.enc1 = Block(in_c,64)
        self.enc2 = Block(64,128)

        self.pool = nn.MaxPool2d(2)

        self.dec1 = Block(128,64)

        self.out  = nn.Conv2d(64,out_c,1)


    def forward(self,x):

        x1 = self.enc1(x)
        x2 = self.pool(x1)

        x3 = self.enc2(x2)

        x4 = F.interpolate(x3,size=x1.shape[-2:],mode="bilinear",align_corners=False)

        x5 = self.dec1(x4)

        return self.out(x5)



# -------- LOAD --------

device="cuda" if torch.cuda.is_available() else "cpu"

ds=WrfVitShardDataset(SHARDS_DIR)

X0,Y0=ds[0]

model=UNet(X0.shape[0],Y0.shape[0]).to(device)

model.load_state_dict(torch.load(CKPT_PATH,map_location=device))

model.eval()


frames=[]


print("Creating GIF frames...")


for i in range(len(ds)):


    X,Y=ds[i]


    xf=X[-1].numpy()

    yf=Y[0].numpy()


    Xb=X.unsqueeze(0).to(device)


    mean=Xb.mean(dim=(0,2,3),keepdim=True)
    std=Xb.std(dim=(0,2,3),keepdim=True)+1e-6

    Xn=(Xb-mean)/std


    with torch.no_grad():

        pred=model(Xn).cpu().squeeze(0)

        pf=torch.sigmoid(pred[0]).numpy()



    true_mask=yf>FIRE_THR

    pred_mask=pf>PB_THR


    inter=(true_mask & pred_mask).sum()

    union=(true_mask | pred_mask).sum()

    iou=inter/union if union>0 else 0


    fig,ax=plt.subplots(1,3,figsize=(15,5))


    fig.suptitle(

        f"Wildfire Prediction Frame {i} | IoU={iou:.3f}",

        fontsize=16

    )


    im0=ax[0].imshow(xf,origin="lower",cmap="inferno")

    ax[0].set_title("Input fire(t)")

    plt.colorbar(im0,ax=ax[0])


    im1=ax[1].imshow(true_mask,origin="lower",cmap="Blues")

    ax[1].set_title("Ground Truth fire(t+K)")


    im2=ax[2].imshow(pf,origin="lower",cmap="magma")

    ax[2].contour(true_mask,colors="cyan",linewidths=1)

    ax[2].contour(pred_mask,colors="lime",linewidths=1)

    ax[2].set_title("Prediction + Contours")


    fig.canvas.draw()


    frame=np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')

    frame=frame.reshape(fig.canvas.get_width_height()[::-1]+(3,))


    frames.append(frame)


    plt.close(fig)



print("Saving GIF...")


imageio.mimsave(

    OUT_GIF,

    frames,

    fps=FPS

)


print("DONE")

print("Saved:",OUT_GIF)
