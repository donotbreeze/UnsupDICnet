from PIL import Image
import flowiz as fz
import matplotlib.pyplot as plt
from Models.UnsupDICnet import *
import numpy as np
from Models.lossFunction import EPE
import cv2

# set the device
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")


if __name__ == "__main__":
    datasize = 256
    # Load label
    flow_path = r"./Images/JHTDB_isotropic1024_hd_00074_flow.flo"
    label = fz.read_flow(flow_path)
    label = torch.FloatTensor([label[..., 0], label[..., 1]])
    flow = label.view(1, 2, datasize, datasize)
    u = label[0].numpy()
    v = label[1].numpy()

    # Load img
    img1_path = r"./Images/JHTDB_isotropic1024_hd_00074_img1.tif"
    img2_path = r"./Images/JHTDB_isotropic1024_hd_00074_img2.tif"
    img1 = torch.FloatTensor(np.asarray(Image.open(img1_path)) * 1.0 / 255.0).view(1, 1, datasize, datasize)
    img2 = torch.FloatTensor(np.asarray(Image.open(img2_path)) * 1.0 / 255.0).view(1, 1, datasize, datasize)

    # Load pretrained model
    weight_save_path = r"./weights/publicDatasetWeights/UnsupDICnet_local1.pt"
    model = UnsupDICnet()
    model.load_state_dict(torch.load(weight_save_path)['model_state_dict'])
    model.eval()
    model.to(device)
    print('load successfully.')

    # show result
    y_pre = estimate(img1.to(device), img2.to(device), model, train=False)
    u_pre = y_pre[0][0].detach().numpy()
    v_pre = y_pre[0][1].detach().numpy()
    u_pre = cv2.medianBulr(u_pre, 5)
    v_pre = cv2.medianBulr(v_pre, 5)

    # AEE
    error = EPE(y_pre, flow).item()
    print(error)

    plt.imshow(u_pre, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    plt.imshow(v_pre, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.show()
