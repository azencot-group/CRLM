import torch.nn as nn
import numpy as np

# This is the main function
def get_rel_depths(model,model_name):
    if 'ResNet' in model_name:
        modules, names, depths, rdepths = getResNetsDepths(model)
    else:
        modules, names, depths, rdepths = getDepths_vgg(model)
    return modules, names, depths, rdepths



def getDepths_vgg(model):
    count = 0
    modules = []
    names = []
    depths = []
    modules.append('input')
    names.append('input')
    depths.append(0)

    for i, module in enumerate(model.features):
        name = module.__class__.__name__
        if 'Conv2d' in name or 'Linear' in name:
            count += 1
        if 'MaxPool2d' in name:
            modules.append(module)
            depths.append(count)
            names.append('MaxPool2d')
    for i, module in enumerate(nn.Sequential(model.classifier)):
        name = module.__class__.__name__
        if 'Linear' in name:
            modules.append(module)
            count += 1
            depths.append(count + 1)
            names.append('Linear')
    depths = np.array(depths)
    rdepths = depths / depths[-1]
    return modules, names, depths, rdepths


def getLayerDepth(layer):
    count = 0
    for m in layer:
        for c in m.children():
            name = c.__class__.__name__
            if 'Conv' in name:
                count += 1
    return count




def getResNetsDepths(model):
    modules = []
    names = []
    depths = []

    # input
    count = 0
    modules.append('input')
    names.append('input')
    depths.append(count)

    # input
    count = 1
    modules.append(model.bn1)
    names.append('bn')
    depths.append(count)

    for module in model.modules():
        name = module.__class__.__name__
        if 'BasicBlock' in name or 'Bottleneck' in name:
            count += getLayerDepth(module.modules())
            modules.append(module)
            names.append(name)
            depths.append(count)

    # average pooling
    count += 1
    modules.append(model.avgpool)
    names.append('avgpool')
    depths.append(count)
    # output
    count += 1
    modules.append(model.linear)
    names.append('Linear')
    depths.append(count)
    depths = np.array(depths)
    rdepths = depths / depths[-1]
    return modules, names, depths, rdepths