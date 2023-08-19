import numpy as np
from PIL import Image
from Models.UnsupDICnet import *
import matplotlib.pyplot as plt
import cv2

# set the device
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

if __name__ == "__main__":
    # WIN -> W 1040 H 1392
    img1_path = r"./Images/pw-0mil-p2-n14-800_0.tif"
    img2_path = r"./Images/pw-0mil-p2-n14-820_0.tif"
    #  plt -> H W C
    img1 = torch.FloatTensor(np.asarray(Image.open(img1_path)) * 1.0 / 255.0).view(1, 1, 1392, 1040)
    img2 = torch.FloatTensor(np.asarray(Image.open(img2_path)) * 1.0 / 255.0).view(1, 1, 1392, 1040)
    img1 = img1[:, :, :, 200:800]
    img2 = img2[:, :, :, 200:800]

    # N C H W
    # Load pretrained model
    weight_save_path = r"./weights/selfbuiltDatasetWeights/UnsupDICnet_selfbuilt.pt"
    model = UnsupDICnet()
    model.load_state_dict(torch.load(weight_save_path)['model_state_dict'])
    model.eval()
    model.to(device)
    print('load successfully.')

    # show prediction
    y_pre = estimate(img1.to(device), img2.to(device), model, train=False)
    v_pre = y_pre[0][1].detach().numpy()
    v_pre = cv2.blur(v_pre, (19, 19))

    plt.imshow(v_pre, cmap='jet', vmin=-3, vmax=1)
    plt.axis('off')
    plt.colorbar()
    plt.show()
