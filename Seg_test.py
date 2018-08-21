import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *
from VOC_dataloader import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from FPN.models.fpn import fpn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--saved_model", type=str, default="5e-1_VOC_BL3_Denoiser_2", help='name of saved model')
# parser.add_argument("--test_data", type=str, default='CBSD68', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

save_dir = 'segmentation_saved_imgs/' + opt.saved_model
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def normalize(data):
    return data/255.

def main():
    print('Loading model ...\n')
    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.saved_model+'.pth')))
    model.eval()

    seg = fpn(21)
    seg = nn.DataParallel(seg).cuda()
    seg.eval()
    seg.load_state_dict(torch.load('logs/latest_net_SateFPN.pth'))

    dataset_val = MultiDataSet(testFlag=True, preload=True, val=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print("# of validation samples: %d\n" % int(len(dataset_val)))

    psnr_val = 0
    ssim_val = 0
    for i, data in enumerate(loader_val, 0):
        # data = data[0]
        img_val = data[0]
        target = data[1]
        noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # if opt.mode == 'S':
        #     noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.noiseL / 255.)
        # if opt.mode == 'B':
        #     noise = torch.zeros(img_val.size())
        #     stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
        #     for n in range(noise.size()[0]):
        #         sizeN = noise[0, :, :, :].size()
        #         noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
        imgn_val = img_val + noise
        img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())

        out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
        psnr_val += batch_PSNR(out_val, img_val, 1.)
        ssim_val += batch_SSIM(out_val, img_val, 1.)



        denoised_img = transforms.ToPILImage()(out_val[0].data.cpu())
        denoised_img.save(os.path.join(save_dir, "denoised_image"+str(i)+'.png'))


        # noisy_img = transforms.ToPILImage()(imgn_val[0].data.cpu())
        # noisy_img.save(os.path.join('save_imgs/Noisy/DnCNN', str(i) + '.png'))
        seg_input = out_val.data.cpu().numpy()
        for n in range(out_val.size()[0]):
            seg_input[n, :, :, :] = demean(seg_input[n, :, :, :])
        seg_input = Variable(torch.from_numpy(seg_input).cuda())
        output = seg(seg_input)

        # Resize target for {100%, 75%, 50%, Max} outputs
        outImg = cv2.resize(output[0].to("cpu").max(0)[1].numpy(), (target.shape[2],target.shape[1]), interpolation=
        cv2.INTER_NEAREST)

        # save_dir = osp.join(CONFIG.SAVE_DIR, "images")
        cv2.imwrite(os.path.join(save_dir, "image" + str(i) + ".png"), cv2.cvtColor(inputImgTransBack(img_val),
                                                                                cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "predict" + str(i) + ".png"), cv2.cvtColor(classToRGB(outImg),
                                                                                  cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "label" + str(i) + ".png"), cv2.cvtColor(classToRGB(target[0].to("cpu")),
                                                                                cv2.COLOR_RGB2BGR))

    psnr_val /= len(loader_val)
    ssim_val /= len(loader_val)

    print("\nPSNR on test data %f" % psnr_val)
    print("\nSSIM on test data %f" % ssim_val)
    print("***********************************************************")



if __name__ == "__main__":
    main()
