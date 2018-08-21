import sys
import os
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import cv2
import numpy as np
import sys

# from utils import *
# from unet import UNet
# sys.path.append("/home/chenwy/isic/pspnet")
from pspnet.pspnet import pspnet
from pspnet.metrics import runningScore
running_metrics = runningScore(2)

from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from tensorboardX import SummaryWriter

#################################
from Denoiser.models import DnCNN
from Denoiser.utils import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# device = torch.device("cuda:6,7" if torch.cuda.is_available() else "cpu")

#################################




def get_batch(TASK1_DATA_DIR, TASK1_TRUTH_DIR, ids):
    # images = [], masks = [], []
    images = []
    masks = []
    for i in range(len(ids)):
        id = ids[i]
        image = cv2.imread(TASK1_DATA_DIR + "ISIC_" + id + ".jpg") # h, w, c
        images.append(image)
        mask = cv2.imread(TASK1_TRUTH_DIR + "ISIC_" + id + "_segmentation.png")[:, :, 2] # h, w
        mask = mask.astype(int)
        mask = mask / 255.
        masks.append(mask)
    return images, masks

def resize(images, shape):
    if len(images[0].shape) == 2:
        resized = np.empty((len(images), shape[1], shape[0]))
    else:
        resized = np.empty((len(images), shape[1], shape[0], 3))
    for i in range(len(images)):
        # print(images[i].shape)
        resized[i] = cv2.resize(images[i] * 1.0, shape)
    return resized

def prepare_images(images, masks):
    # b, h, w, c AND b, h, w
    images = np.swapaxes(images, 1, 3) # b, c, w, h
    masks = np.swapaxes(masks, 1, 2) # b, w, h
    images = Variable(torch.Tensor(images)).cuda()
    masks = Variable(torch.Tensor(masks)).cuda()
    return images, masks

def restore_shape(predictions, images):
    '''
    predictions: b, w, h
    images: b, h, w, c
    '''
    restored = []
    for i in range(len(predictions)):
        restored.append(np.swapaxes(np.round(cv2.resize(predictions[i] * 1.0, images[i].shape[:2])).astype(int), 0, 1)) # h, w
    return restored


def crop(images, masks, predictions):
    # h, w, c
    images_small, masks_small = np.empty(len(images), dtype=object), np.empty(len(images), dtype=object)
    # images_small = np.empty(len(images), dtype=object)
    margins = np.empty(len(images), dtype=object)
    for i in range(len(images)):
        image, mask, prediction = images[i], masks[i], predictions[i]
        # image, prediction = images[i], predictions[i]
        top, bottom, left, right = 0, -1, 0, -1
        if prediction.sum() < 1:
            images_small[i] = image
            masks_small[i] = mask
        else:
            while prediction[top, :].sum() < 1: top += 1
            while prediction[bottom, :].sum() < 1: bottom -= 1
            while prediction[:, left].sum() < 1: left += 1
            while prediction[:, right].sum() < 1: right -= 1
            top = int(np.ceil(top * 0.9))
            bottom = int(np.floor(bottom * 0.9))
            left = int(np.ceil(left * 0.9))
            right = int(np.floor(right * 0.9))
            images_small[i] = image[top: image.shape[0] + bottom + 1, left: image.shape[1] + right + 1]
            # print(prediction.shape, image.shape, top, bottom, left, right, images_small[i].shape)
            masks_small[i] = mask[top: image.shape[0] + bottom + 1, left: image.shape[1] + right + 1]
        margins[i] = (top, - bottom - 1, left, - right - 1)
    return images_small, masks_small, margins


def padding(predictions_small, margins):
    predictions = []
    for i in range(len(predictions_small)):
        predictions.append(np.pad(predictions_small[i], [(margins[i][0], margins[i][1]), (margins[i][2], margins[i][3])], mode='constant', constant_values=0))
    return predictions

def prepare_for_denoising(images):
    images = torch.unsqueeze(torch.Tensor(images / 255.), 0)
    images = np.swapaxes(images, 1, 3)  # b, c, w, h
    return images



# task_name = "pspnet_6_29_2018"
# task_name = "pspnet_2step_7_13_2018"


DATA_DIR = "/ssd1/chenwy/isic/"
TASK1_DATA_DIR = DATA_DIR + 'ISIC2018_Task1-2_Training_Input/'
TASK1_TRUTH_DIR = DATA_DIR + 'ISIC2018_Task1_Training_GroundTruth/'
TASK1_VAL_DIR = DATA_DIR + 'ISIC2018_Task1-2_Validation_Input/'
TASK1_TEST_DIR = DATA_DIR + 'ISIC2018_Task1-2_Test_Input/'

ids_all = [ img[5:-4] for img in os.listdir(TASK1_DATA_DIR) if img.endswith("jpg") ]
ids_train, ids_test = train_test_split(ids_all, test_size=0.1, random_state=42)
# ids_train, ids_val = train_test_split(ids_train, test_size=0.1, random_state=42)

# ids_all = [ img[5:-4] for img in os.listdir(TASK1_VAL_DIR) if img.endswith("jpg") ]

print(len(ids_all))
print(len(ids_train))
# print(len(ids_val))
print(len(ids_test))

batch_size = 2
cuda = True
pretrain = True

# size1 = (100, 150)
size1 = (300, 300)
size2 = (300, 300)

if cuda:
    # psp1 = pspnet(n_classes=2, input_size=size1).cuda()
    # psp2 = pspnet(n_classes=2, input_size=size2).cuda()
    # denoiser = DnCNN(channels=3, num_of_layers=20).cuda()
    psp1 = pspnet(n_classes=2, input_size=size1).to('cuda:0')
    psp2 = pspnet(n_classes=2, input_size=size2).to('cuda:0')
    denoiser = DnCNN(channels=3, num_of_layers=20).to('cuda:0')
else:
    model = UNet(3, 2)




# psp1 = psp1.cuda()
# psp1 = nn.DataParallel(psp1)
# psp1.load_state_dict(torch.load("/home/chenwy/isic/saved_models/" + task_name + ".pth"))
# psp1 = psp1.cuda()
psp1 = nn.DataParallel(psp1).to('cuda:0')
psp1.load_state_dict(torch.load("/home/chenwy/isic/saved_models/" + "pspnet_2step_7_24_2018_1" + ".pth"))
# psp2 = psp2.cuda()
psp2 = nn.DataParallel(psp2).to('cuda:0')
psp2.load_state_dict(torch.load("/home/chenwy/isic/saved_models/" + "pspnet_2step_7_24_2018_2" + ".pth"))
# psp1.eval()
# psp2.eval()
denoiser_model = 'VOC_BL1_Denoiser'
noise_sigma = 35
print (denoiser_model)
print('sigma = ', noise_sigma)
print (denoiser_model)
denoiser = nn.DataParallel(denoiser).to('cuda:0')
denoiser.load_state_dict(torch.load('Saved_Denoiser/' + denoiser_model + '.pth'))
denoiser.eval()

print("evaluating...")
# for i in tqdm(range(0, int(np.ceil(len(ids_test) * 1.0 / batch_size)))):
#     ids = ids_test[i * batch_size: (i + 1) * batch_size]
#     # print(ids)
#     images, masks = get_batch(TASK1_DATA_DIR, TASK1_TRUTH_DIR, ids, cuda) # h, w, c
#     
#     images_var, masks_var = resize(images, size1), resize(masks, size1)
#     images_var, masks_var = prepare_images(images_var, masks_var)
#     _, outputs = psp1(images_var) # b, c = 2, w, h
#     
#     outputs = F.softmax(outputs, dim=1)
#     
#     predictions = outputs.argmax(1).data.cpu().numpy() # b, w, h
#     predictions = restore_shape(predictions, masks) # b, h, w
#     
#     images_small, masks_small, margins = crop(images, masks, predictions)
#     images_var, masks_var = resize(images_small, size2), resize(masks_small, size2)
#     images_var, masks_var = prepare_images(images_var, masks_var)
#     _, outputs = psp2(images_var)
#     predictions_small = outputs.argmax(1).data.cpu().numpy() # b, w, h
#     predictions_small = restore_shape(predictions_small, masks_small)
#     predictions = padding(predictions_small, margins)
#     
#     for j in range(batch_size):
#         shape = predictions[j].shape
#         if predictions[j].sum() * 1.0 / (shape[0] * shape[1]) < 0.003:
#             print(predictions[j].sum() * 1.0 / (shape[0] * shape[1]))
#             center = (shape[0] // 2, shape[1] // 2)
#             r = np.min(shape) * 0.24
#             for x in range(shape[0]):
#                 for y in range(shape[1]):
#                     if ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5 <= r: predictions[j][x, y] = 1
#             print(predictions[j].sum() * 1.0 / (shape[0] * shape[1]))
#     
#     running_metrics.update(masks, predictions)
# 
# score, class_iou = running_metrics.get_scores()
# iou_val, score_val = score["IoU"], score["IoU_threshold"]
# running_metrics.reset()
# 
# print('IoU:{:.4f}, score:{:.4f}'.format(iou_val, score_val))



############################

###########################
# noise_sigma = 25
# print('sigma = ', noise_sigma)
# psnr_test = 0
# ssim_test = 0
psnr_test = []
ssim_test = []
# IOU = 0
############################
for i in tqdm(range(0, int(np.ceil(len(ids_test) * 1.0 / batch_size)))):
    with torch.no_grad():
        ids = ids_test[i * batch_size: (i + 1) * batch_size]
        images_, masks = get_batch(TASK1_DATA_DIR, TASK1_TRUTH_DIR, ids) # h, w, c   # task1 data dir = training input


        ############################
        # add noise to clean images
        images = []
        for j in range(batch_size):
            resize_img = cv2.resize(images_[j], (1000, 1000), interpolation=cv2.INTER_CUBIC)
            masks[j] = cv2.resize(masks[j], (1000, 1000), interpolation=cv2.INTER_CUBIC)
            ISource = prepare_for_denoising(resize_img)
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std= noise_sigma/ 255.)
            INoisy = ISource + noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            Out = torch.clamp(INoisy - denoiser(INoisy), 0., 1.)
            psnr = batch_PSNR(Out, ISource, 1.)
            # psnr_test += psnr
            ssim = batch_SSIM(Out, ISource, 1.)
            # ssim_test += ssim
            im = np.squeeze(np.swapaxes(255.*(Out.data.cpu().numpy()), 1, 3))
            images.append(im)
            psnr_test.append(psnr)
            ssim_test.append(ssim)

        ############################


        images_var, masks_var = resize(images, size1), resize(masks, size1)
        images_var, masks_var = prepare_images(images_var, masks_var)
        outputs = psp1(images_var) # b, c = 2, w, h
        predictions = outputs[1].argmax(1).data.cpu().numpy() # b, w, h
        # predictions = outputs.argmax(1).data.cpu().numpy()  # b, w, h
        predictions = restore_shape(predictions, images) # b, h, w

        images_small, masks_small, margins = crop(images, masks, predictions)
        images_var, masks_var = resize(images_small, size2), resize(masks_small, size2)
        images_var, masks_var = prepare_images(images_var, masks_var)
        outputs = psp2(images_var)
        predictions_small = outputs[1].argmax(1).data.cpu().numpy()  # b, w, h
        # predictions_small = outputs.argmax(1).data.cpu().numpy() # b, w, h
        predictions_small = restore_shape(predictions_small, images_small)
        predictions = padding(predictions_small, margins)

        for j in range(batch_size):
            shape = predictions[j].shape
            if predictions[j].sum() * 1.0 / (shape[0] * shape[1]) < 0.003:
                print(predictions[j].sum() * 1.0 / (shape[0] * shape[1]))
                center = (shape[0] // 2, shape[1] // 2)
                r = np.min(shape) * 0.24
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        if ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5 <= r: predictions[j][x, y] = 1
                print(predictions[j].sum() * 1.0 / (shape[0] * shape[1]))

        running_metrics.update(masks, predictions)

        score, class_iou = running_metrics.get_scores()
        iou_val, score_val = score["IoU_batch"], score["IoU_threshold_batch"]
        for j in range(batch_size):
            if iou_val[j] < 0.65:
                print(ids[j], iou_val[i * batch_size + j], psnr_test[i * batch_size + j], ssim_test[i * batch_size + j])
                # print(ids[j], iou_val[j], psnr_test[j], ssim[j])
        #         ############
        #         # save denoised image
        #         ############
        #         cv2.imwrite('BL3_Denoised/' + str(ids[j]) + "_prediction" + str(iou_val[j]) + ".png", predictions[j] * 255.)
        #         cv2.imwrite('BL3_Denoised/' + str(ids[j]) + "_denoised.png", images[j])
        #         cv2.imwrite('BL3_Denoised/' + str(ids[j]) + "_mask.png", masks[j] * 255.)
        #         # IOU += iou_val[j]
            else:
                print(ids[j], score_val[i * batch_size + j], psnr_test[i * batch_size + j], ssim_test[i * batch_size + j])
        #         # print(ids[j], score_val[j], psnr, ssim)
        #         ############
        #         # save denoised image
        #         ############
        #         cv2.imwrite('BL3_Denoised/' + str(ids[j]) + "_prediction.png", predictions[j] * 255.)
        #         cv2.imwrite('BL3_Denoised/' + str(ids[j]) + "_denoised.png", images[j])
        #         cv2.imwrite('BL3_Denoised/' + str(ids[j]) + "_mask.png", masks[j] * 255.)

        # running_metrics.reset()
        print('Average IoU_Val =', sum(iou_val)/float(len(iou_val)))
        print('Average PSNR =', sum(psnr_test)/float(len(psnr_test)))
        print('Average SSIM =', sum(ssim_test)/float(len(ssim_test)))




# for i in tqdm(range(0, int(np.ceil(len(ids_all) * 1.0 / batch_size)))):
#     ids = ids_all[i * batch_size: (i + 1) * batch_size]
#     images = get_batch(TASK1_VAL_DIR, TASK1_TRUTH_DIR, ids, cuda) # h, w, c
#     
#     images_var = resize(images, size1)
#     images_var = prepare_images(images_var)
#     outputs = psp1(images_var) # b, c = 2, w, h
#     predictions = outputs[1].argmax(1).data.cpu().numpy() # b, w, h
#     predictions = restore_shape(predictions, images) # b, h, w
#     
#     for j in range(batch_size):
#         shape = predictions[j].shape
#         if predictions[j].sum() * 1.0 / (shape[0] * shape[1]) < 0.003:
#             print(predictions[j].sum() * 1.0 / (shape[0] * shape[1]))
#             center = (shape[0] // 2, shape[1] // 2)
#             r = np.min(shape) * 0.24
#             for x in range(shape[0]):
#                 for y in range(shape[1]):
#                     if ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5 <= r: predictions[j][x, y] = 1
#             print(predictions[j].sum() * 1.0 / (shape[0] * shape[1]))
#     
#     images_small, margins = crop(images, predictions)
#     images_var = resize(images_small, size2)
#     images_var = prepare_images(images_var)
#     outputs = psp2(images_var)
#     predictions_small = outputs[1].argmax(1).data.cpu().numpy() # b, w, h
#     predictions_small = restore_shape(predictions_small, images_small)
#     predictions = padding(predictions_small, margins)
#     
#     for j in range(batch_size):
#         shape = predictions[j].shape
#         if predictions[j].sum() * 1.0 / (shape[0] * shape[1]) < 0.003:
#             print(predictions[j].sum() * 1.0 / (shape[0] * shape[1]))
#             center = (shape[0] // 2, shape[1] // 2)
#             r = np.min(shape) * 0.24
#             for x in range(shape[0]):
#                 for y in range(shape[1]):
#                     if ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5 <= r: predictions[j][x, y] = 1
#             print(predictions[j].sum() * 1.0 / (shape[0] * shape[1]))
#     
#     for j in range(batch_size):
#         cv2.imwrite(str(ids[j]) + "_segmentation" + ".png", predictions[j] * 255.)
