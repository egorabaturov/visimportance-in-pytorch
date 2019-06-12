import os
import numpy as np
import PIL.Image
from scipy.special import expit
import torch
from torch.autograd import Variable
import models
import utils

img_dir = './examples/in'
img_name = "fabuza.png"
pretrained_model = './pretrained_models/gdi_fcn16.pth'
save_dir = './examples/out'
fcn_type = 'fcn16'  # 'fcn32', 'fcn16'

def load_img(img_path):
    img = PIL.Image.open(img_path)
    img = np.array(img, dtype=np.uint8)

    return img

def transform_img(img):
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])

    img_copy = img.copy()
    img_copy = img_copy[:, :, ::-1]  # RGB -> BGR
    img_copy = img_copy.astype(np.float32)
    img_copy -= mean_bgr
    img_copy = img_copy.transpose(2, 0, 1)  # C x H x W
    img_copy = torch.from_numpy(img_copy).float()

    return img_copy

img_path = os.path.join(img_dir, img_name)
img = load_img(img_path)
img_transformed = transform_img(img)[np.newaxis]
model = models.FCN32s() if fcn_type == 'fcn32' else models.FCN16s()
model_weight = torch.load(pretrained_model)
model.load_state_dict(model_weight)
with torch.no_grad():
    img_transformed = Variable(img_transformed)
score = model(img_transformed)
lbl_pred = (expit(score.data.cpu().numpy()) * 255).astype(np.uint8)[0][0]
save_path = os.path.join(save_dir, 'test_' + img_name)
utils.overlay_imp_on_img(img, lbl_pred, save_path, colormap='jet')
