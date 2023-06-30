##########
# ### Import Packages and Setup Paths

# import base packages
import os
from PIL import Image
import numpy as np
import torch
import random
import gradio as gr

# import packages for image decomposition
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import gdown

# import packages for image+background composition
import importlib
from omegaconf import OmegaConf

# project level imports
from data_loader_cache import normalize, im_reader, im_preprocess
from models import *

# use gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# download model if not already downloaded
if not os.path.exists("saved_models"):
    os.mkdir("saved_models")
    MODEL_PATH_URL = "https://drive.google.com/uc?id=1KyMpRjewZdyYfxHPYcd-ZbanIXtin0Sn"
    gdown.download(MODEL_PATH_URL, "saved_models/isnet.pth", use_cookies=False)

# dependency functions to construct image decomposition model

# normalize the image
class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image

# intialize the transform object
transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])

# load the image and transform
def load_image(im_path, hypar):
    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)  # make a batch of image, shape

# construct the main decomposition net
def build_model(hypar, device):
    net = hypar["model"]  # GOSNETINC(3,1)

    # convert to half precision
    if (hypar["model_digit"] == "half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if (hypar["restore_model"] != ""):
        net.load_state_dict(torch.load(hypar["model_path"] + "/" + hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()
    return net

# generate the image mask
def predict(net, inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if (hypar["model_digit"] == "full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)  # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0]  # list of 6 results

    pred_val = ds_val[0][0, :, :, :]  # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(
        # F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))
        F.interpolate(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)  # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)  # it is the mask we need

# use the default hyperparameters and construct the model
hypar = {}  # paramters for inferencing

hypar["model_path"] = "./saved_models"  ## load trained weights from this path
hypar["restore_model"] = "isnet.pth"  ## name of the to-be-loaded weights
hypar["interm_sup"] = False  ## indicate if activate intermediate feature supervision

##  choose floating point accuracy --
hypar["model_digit"] = "full"  ## indicates "half" or "full" accuracy of float number
hypar["seed"] = 0

hypar["cache_size"] = [1024, 1024]  ## cached input spatial resolution, can be configured into different size

## data augmentation parameters ---
hypar["input_size"] = [1024,
                       1024]  ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
hypar["crop_size"] = [1024,
                      1024]  ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation

hypar["model"] = ISNetDIS()

# construct the model
net = build_model(hypar, device)

# inference function to generate decomposed image and mask
def inference(image: str):
    image_path = image

    image_tensor, orig_size = load_image(image_path, hypar)
    mask = predict(net, image_tensor, orig_size, hypar, device)

    pil_mask = Image.fromarray(mask).convert('L')
    im_rgb = Image.open(image).convert("RGB")

    im_rgba = im_rgb.copy()
    im_rgba.putalpha(pil_mask)

    return (im_rgba, pil_mask)

# configure image recomposition
config = OmegaConf.load("config/annotator.yaml")
package_annotator = "processor"
cls, input_element = config["alpha"]["process"], config["alpha"].get("input")

# function to process and recompose the image
def process_image(cls: str, fg: Image.Image, bg: Image.Image, *kwargs):
    if fg.size != bg.size:
        fg = fg.resize(bg.size)
    module_imp = importlib.import_module(package_annotator)
    model = getattr(module_imp, cls)
    image_processor = model()
    result = image_processor(fg, bg, *kwargs)
    if type(result) == tuple:
        return result
    return [result]

# function to retrieve list of file names
def get_file_list(file_path: str) -> list:
    # initialize the list to return
    file_list = list()
    
    # load in all file names as a list
    for (dirpath, dirnames, filenames) in os.walk(file_path):
        file_list.extend(filenames)
    
    # return list of files
    return file_list

# main function to create file
def predict_img(image_0):
    # INGEST HERE 
    # image_path = os.path.join(test_images, "test_0.jpg")
    im_fg, im_mask = inference(image_0)
    
    # get dimensions of foreground image
    fg_w = im_fg.width
    fg_h = im_fg.height

    # get background images
    background_path = r"data/bg"
    background_image_list = get_file_list(os.path.join(background_path))
    print(background_image_list)
    
    # crop or resize background
    img_bg = Image.open(os.path.join(background_path, random.choice(background_image_list)))
    img_bg_adj = img_bg.convert("RGB").resize((fg_w, fg_h)) # can also crop from center
    
    # overlay foreground on background
    img_comp = process_image(cls=cls, fg=im_fg, bg=img_bg_adj)
    
    return img_comp[0]

gr.Interface(
    predict_img,
    inputs=[gr.inputs.Image(label="Upload User Photo", type="filepath")],
    outputs=gr.inputs.Image(label="Output Photo", type="filepath"),
    title="Image Background Sticker",
).launch()
# gr.Interface(
#     predict_img,
#     inputs=[gr.inputs.Image(label="Upload User Photo", type="filepath"), 
#             gr.inputs.Image(label="Upload Background", type="filepath")],
#     outputs=gr.inputs.Image(label="Output Photo", type="filepath"),
#     title="Image Background Sticker",
# ).launch()