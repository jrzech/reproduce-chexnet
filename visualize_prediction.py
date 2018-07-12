from __future__ import print_function, division

# pytorch imports
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms, utils

# image / graphics imports
from skimage import io, transform
from PIL import Image
from pylab import *
import seaborn as sns
from matplotlib.pyplot import show 

# data science
import numpy as np
import scipy as sp
import pandas as pd

# import other modules
from copy import deepcopy
import cxr_dataset as CXR
import eval_model as E

def calc_cam(x, label, model):
    """
    function to generate a class activation map corresponding to a torch image tensor

    Args:
        x: the 1x3x224x224 pytorch tensor file that represents the NIH CXR
        label:user-supplied label you wish to get class activation map for; must be in FINDINGS list
        model: densenet121 trained on NIH CXR data

    Returns:
        cam_torch: 224x224 torch tensor containing activation map
    """
    FINDINGS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia']

    if label not in FINDINGS:
        raise ValueError(
            str(label) +
            "is an invalid finding - please use one of " +
            str(FINDINGS))

    # find index for label; this corresponds to index from output of net
    label_index = next(
        (x for x in range(len(FINDINGS)) if FINDINGS[x] == label))

    # define densenet_last_layer class so we can get last 1024 x 7 x 7 output
    # of densenet for class activation map
    class densenet_last_layer(torch.nn.Module):
        def __init__(self, model):
            super(densenet_last_layer, self).__init__()
            self.features = torch.nn.Sequential(
                *list(model.children())[:-1]
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.nn.functional.relu(x, inplace=True)
            return x

    # instantiate cam model and get output
    model_cam = densenet_last_layer(model)
    x = torch.autograd.Variable(x)
    y = model_cam(x)
    y = y.cpu().data.numpy()
    y = np.squeeze(y)

    # pull weights corresponding to the 1024 layers from model
    weights = model.state_dict()['classifier.0.weight']
    weights = weights.cpu().numpy()
    
    bias = model.state_dict()['classifier.0.bias']
    bias = bias.cpu().numpy()
    
    # can replicate bottleneck and probability calculation here from last_layer network and params from
    # original network to ensure that reconstruction is accurate -- commented out as previously checked
    
    #model_bn = deepcopy(model)
    #new_classifier = torch.nn.Sequential(*list(model_bn.classifier.children())[:-2])
    #model_bn.classifier = new_classifier
    #bn=model_bn(x)
    #recreate=0
    #bottleneck = []
    #for k in range(0,1024):
    #    avg_value = np.mean(y[k,:,:])# over the 7x7 grid
    #    bottleneck.append(avg_value)
    #    recreate = recreate+weights[label_index,k]*avg_value
    #recreate = recreate + bias[label_index]
    #recreate = 1/(1+math.exp(-recreate))
    #print("recalc:")
    #print(recreate)
    #print("original:")
    #print(model(x).data.numpy()[0][label_index])

    # create 7x7 cam
    cam = np.zeros((7, 7, 1))
    for i in range(0, 7):
        for j in range(0, 7):
            for k in range(0, 1024):
                cam[i, j] += y[k, i, j] * weights[label_index, k]
    cam+=bias[label_index]

    #make cam into local region probabilities with sigmoid
    
    cam=1/(1+np.exp(-cam))
    
    label_baseline_probs={
        'Atelectasis':0.103,
        'Cardiomegaly':0.025,
        'Effusion':0.119,
        'Infiltration':0.177,
        'Mass':0.051,
        'Nodule':0.056,
        'Pneumonia':0.012,
        'Pneumothorax':0.047,
        'Consolidation':0.042,
        'Edema':0.021,
        'Emphysema':0.022,
        'Fibrosis':0.015,
        'Pleural_Thickening':0.03,
        'Hernia':0.002
    }
    
    #normalize by baseline probabilities
    cam = cam/label_baseline_probs[label]
    
    #take log
    cam = np.log(cam)
    
    return cam

def load_data(
        PATH_TO_IMAGES,
        LABEL,
        PATH_TO_MODEL,
        POSITIVE_FINDINGS_ONLY,
        STARTER_IMAGES):
    """
    Loads dataloader and torchvision model

    Args:
        PATH_TO_IMAGES: path to NIH CXR images
        LABEL: finding of interest (must exactly match one of FINDINGS defined below or will get error)
        PATH_TO_MODEL: path to downloaded pretrained model or your own retrained model
        POSITIVE_FINDINGS_ONLY: dataloader will show only examples + for LABEL pathology if True, otherwise shows positive
                                and negative examples if false

    Returns:
        dataloader: dataloader with test examples to show
        model: fine tuned torchvision densenet-121
    """

    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    del checkpoint
    model.cpu()

    # build dataloader on test
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    FINDINGS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia']

    data_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if not POSITIVE_FINDINGS_ONLY:
        finding = "any"
    else:
        finding = LABEL

    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='test',
        transform=data_transform,
        finding=finding,
        starter_images=STARTER_IMAGES)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1)
    
    return iter(dataloader), model


def show_next(dataloader, model, LABEL):
    """
    Plots CXR, activation map of CXR, and shows model probabilities of findings

    Args:
        dataloader: dataloader of test CXRs
        model: fine-tuned torchvision densenet-121
        LABEL: finding we're interested in seeing heatmap for
    Returns:
        None (plots output)
    """
    FINDINGS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia']
    
    label_index = next(
        (x for x in range(len(FINDINGS)) if FINDINGS[x] == LABEL))

    # get next iter from dataloader
    try:
        inputs, labels, filename = next(dataloader)
    except StopIteration:
        print("All examples exhausted - rerun cells above to generate new examples to review")
        return None
        
    # get cam map
    original = inputs.clone()
    raw_cam = calc_cam(inputs, LABEL, model)
    
    # create predictions for label of interest and all labels
    pred = model(torch.autograd.Variable(original.cpu())).data.numpy()[0]
    predx = ['%.3f' % elem for elem in list(pred)]
    
    fig, (showcxr,heatmap) =plt.subplots(ncols=2,figsize=(14,5))
    
    hmap = sns.heatmap(raw_cam.squeeze(),
            cmap = 'viridis',
            alpha = 0.3, # whole heatmap is translucent
            annot = True,
            zorder = 2,square=True,vmin=-5,vmax=5
            )
    
    cxr=inputs.numpy().squeeze().transpose(1,2,0)    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cxr = std * cxr + mean
    cxr = np.clip(cxr, 0, 1)
        
    hmap.imshow(cxr,
          aspect = hmap.get_aspect(),
          extent = hmap.get_xlim() + hmap.get_ylim(),
          zorder = 1) #put the map under the heatmap
    hmap.axis('off')
    hmap.set_title("P("+LABEL+")="+str(predx[label_index]))
    
    showcxr.imshow(cxr)
    showcxr.axis('off')
    showcxr.set_title(filename[0])
    plt.savefig(str(LABEL+"_P"+str(predx[label_index])+"_file_"+filename[0]))
    plt.show()
    
    
        
    preds_concat=pd.concat([pd.Series(FINDINGS),pd.Series(predx),pd.Series(labels.numpy().astype(bool)[0])],axis=1)
    preds = pd.DataFrame(data=preds_concat)
    preds.columns=["Finding","Predicted Probability","Ground Truth"]
    preds.set_index("Finding",inplace=True)
    preds.sort_values(by='Predicted Probability',inplace=True,ascending=False)
    
    return preds
