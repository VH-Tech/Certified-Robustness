from archs import *
from archs.dncnn import DnCNN
from archs.cifar_resnet import resnet as resnet_cifar
from archs.memnet import MemNet
from archs.wrn import WideResNet
from datasets import get_normalize_layer, get_input_center_layer
from torch.nn.functional import interpolate
from torchvision.models.resnet import resnet18, resnet34
from archs.resnet import resnet50
from archs.swin_transformer import swin_base_patch4_window7_224
# from Convnets.Conv-Adapter.models.backbones.resnet import resnet50
# from Convnets.conv_adapter.models.backbones.resnet import resnet50, resnet18, resnet34
from datasets import get_num_classes
from transformers import ViTForImageClassification, AutoModelForImageClassification

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, inplanes, outplanes, dropout):
        super().__init__()
        self.linear = nn.Linear(inplanes, outplanes)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x
    

IMAGENET_CLASSIFIERS = [
                        'resnet18', 
                        'resnet34', 
                        'resnet50',
                        "google/vit-base-patch16-224-in21k",
                        "vit", "vit_custom"
                        ]

CIFAR10_CLASSIFIERS = [
                        'cifar_resnet110', 
                        'cifar_wrn', 'cifar_wrn40',
                        'VGG16', 'VGG19', 'ResNet18','PreActResNet18','GoogLeNet',
                        'DenseNet121','ResNeXt29_2x64d','MobileNet','MobileNetV2',
                        'SENet18','ShuffleNetV2','EfficientNetB0'
                        'imagenet32_resnet110', 'imagenet32_wrn',"vit","swin"
                        ]

CLASSIFIERS_ARCHITECTURES = IMAGENET_CLASSIFIERS + CIFAR10_CLASSIFIERS

DENOISERS_ARCHITECTURES = ["cifar_dncnn", "cifar_dncnn_wide", "memnet", # cifar10 denoisers
                            'imagenet_dncnn', 'imagenet_memnet' # imagenet denoisers
                        ]

def get_architecture(arch: str, dataset: str, pytorch_pretrained: bool=False, tuning_method: str = "full") -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    ## ImageNet classifiers
    if arch == "resnet18" and dataset in ["pneumonia", "breakhis", "isic", "hyper"]:
        model = resnet18(pretrained=pytorch_pretrained)
        model.fc = nn.Linear(512, get_num_classes(dataset))
        model = model.cuda()
    elif arch == "resnet34" and dataset in ["pneumonia", "breakhis", "isic", "hyper"]:
        model = resnet34(pretrained=pytorch_pretrained)
        model.fc = nn.Linear(512, get_num_classes(dataset))

    elif arch == "resnet50" :
        tuning_config = {"method" : tuning_method, "adapt_size" : 8, "adapt_scale" : 1.0, "kernel_size" : 3}
        model = resnet50(pretrained=True, num_classes=get_num_classes(dataset), tuning_config=tuning_config)
        model.head = LinearHead(model.num_features, get_num_classes(dataset), 0.2)# nn.Linear(model.num_features, num_classes)

        # freeze parameters if needed
        if tuning_method == 'full':
            # all parameters are trainable
            pass 
        elif tuning_method == 'prompt':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue

                if name.startswith('norm'):
                    continue

                if 'tuning_module' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'adapter':
            raise NotImplementedError
        elif tuning_method == 'sidetune':
            raise NotImplementedError
        elif tuning_method == 'linear':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if name.startswith('norm'):
                    continue

                param.requires_grad = False
        elif tuning_method == 'norm':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue

                if 'bn' in name:
                    continue

                if 'gn' in name:
                    continue

                if 'norm' in name:
                    continue
                
                # adjust last group norm
                if 'before_head' in name:
                    continue

                param.requires_grad = False    
        elif tuning_method == 'bias':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if name.startswith('norm'):
                    continue

                if 'bias' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'conv_adapt' or tuning_method == 'repnet':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue
                
                # add a norm layer before average pooling
                if 'norm' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'conv_adapt_norm':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue

                if 'bn' in name:
                    continue

                if 'gn' in name:
                    continue

                if 'norm' in name:
                    continue
                
                # adjust last group norm
                if 'before_head' in name:
                    continue

                param.requires_grad = False    
        elif tuning_method == 'conv_adapt_bias' or tuning_method == 'repnet_bias':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue

                if 'bias' in name:
                    continue
                
                # add a norm layer before average pooling
                if name.startswith('norm'):
                    continue

                param.requires_grad = False

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")

        model = model.cuda()
    #ViT
    elif arch == "vit" and dataset in ["pneumonia", "breakhis", "isic", "hyper"]:

        # Specify the cache directory
        cache_directory = './model_cache'

        # Load the model, specifying the cache directory
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=get_num_classes(dataset), cache_dir=cache_directory)
        model = model.cuda()

    elif arch == "vit" and dataset in ["cifar10"]:

        # Specify the cache directory
        cache_directory = './model_cache'

        # Load the model, specifying the cache directory
        model = ViTForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", num_labels=get_num_classes(dataset), cache_dir=cache_directory)
        model = model.cuda()

    elif arch == "vit_custom":
        model = VisionTransformer(config=CONFIGS['ViT-B_16'], img_size=224, num_classes=get_num_classes(dataset), zero_head=True, tuning_mode=tuning_method)
        if tuning_method == 'full':
            model.load_from(np.load("archs/weights/ViT-B_16-224.npz"), strict=False)

    #Swin
    elif arch == "swin" :
        tuning_config = {"method" : tuning_method, "adapt_size" : 8, "adapt_scale" : 1.0, "kernel_size" : 3}
        model = swin_base_patch4_window7_224(pretrained=True, num_classes=get_num_classes(dataset), tuning_config=tuning_config)
        model.head = LinearHead(model.num_features, get_num_classes(dataset), 0.2)# nn.Linear(model.num_features, num_classes)

        # freeze parameters if needed
        if tuning_method == 'full':
            # all parameters are trainable
            pass 
        elif tuning_method == 'prompt':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue

                if name.startswith('norm'):
                    continue

                if 'tuning_module' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'adapter':
            raise NotImplementedError
        elif tuning_method == 'sidetune':
            raise NotImplementedError
        elif tuning_method == 'linear':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if name.startswith('norm'):
                    continue

                param.requires_grad = False
        elif tuning_method == 'norm':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue

                if 'bn' in name:
                    continue

                if 'gn' in name:
                    continue

                if 'norm' in name:
                    continue
                
                # adjust last group norm
                if 'before_head' in name:
                    continue

                param.requires_grad = False    
        elif tuning_method == 'bias':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if name.startswith('norm'):
                    continue

                if 'bias' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'conv_adapt' or tuning_method == 'repnet':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue
                
                # add a norm layer before average pooling
                if 'norm' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'conv_adapt_norm':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue

                if 'bn' in name:
                    continue

                if 'gn' in name:
                    continue

                if 'norm' in name:
                    continue
                
                # adjust last group norm
                if 'before_head' in name:
                    continue

                param.requires_grad = False    
        elif tuning_method == 'conv_adapt_bias' or tuning_method == 'repnet_bias':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue

                if 'bias' in name:
                    continue
                
                # add a norm layer before average pooling
                if name.startswith('norm'):
                    continue

                param.requires_grad = False
        elif tuning_method == 'compacter':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue
                param.requires_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")

        model = model.cuda()

    # elif arch == "swin":
    #     # Specify the cache directory
    #     cache_directory = './model_cache'

    #     # load pretrained model
    #     model = AutoModelForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224",  cache_dir=cache_directory)
    #     # Update the final layer to match the number of CIFAR-10 classes (10)
    #     model.config.num_labels = get_num_classes(dataset)
    #     model.classifier = torch.nn.Linear(model.config.hidden_size, model.config.num_labels)
    #     model = model.cuda()

    elif arch == "resnet18" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet18(pretrained=pytorch_pretrained)).cuda()
        cudnn.benchmark = True
    elif arch == "resnet34" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet34(pretrained=pytorch_pretrained)).cuda()
        cudnn.benchmark = True
    elif arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=pytorch_pretrained)).cuda()
        cudnn.benchmark = True

    ## Cifar classifiers
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "imagenet32_resnet110":
        model = resnet_cifar(depth=110, num_classes=1000).cuda()
    elif arch == "imagenet32_wrn":
        model = WideResNet(depth=28, num_classes=1000, widen_factor=10).cuda()

    # Cifar10 Models from https://github.com/kuangliu/pytorch-cifar
    # The 14 models we use in the paper as surrogate models 
    elif arch == "cifar_wrn":
        model = WideResNet(depth=28, num_classes=10, widen_factor=10).cuda()
    elif arch == "cifar_wrn40":
        model = WideResNet(depth=40, num_classes=10, widen_factor=10).cuda()
    elif arch == "VGG16":
        model = VGG('VGG16').cuda()
    elif arch == "VGG19":
        model = VGG('VGG19').cuda()
    elif arch == "ResNet18":
        model = ResNet18().cuda()
    elif arch == "PreActResNet18":
        model = PreActResNet18().cuda()
    elif arch == "GoogLeNet":
        model = GoogLeNet().cuda()
    elif arch == "DenseNet121":
        model = DenseNet121().cuda()
    elif arch == "ResNeXt29_2x64d":
        model = ResNeXt29_2x64d().cuda()
    elif arch == "MobileNet":
        model = MobileNet().cuda()
    elif arch == "MobileNetV2":
        model = MobileNetV2().cuda()
    elif arch == "SENet18":
        model = SENet18().cuda()
    elif arch == "ShuffleNetV2":
        model = ShuffleNetV2(1).cuda()
    elif arch == "EfficientNetB0":
        model = EfficientNetB0().cuda()

    ## Image Denoising Architectures
    elif arch == "cifar_dncnn":
        model = DnCNN(image_channels=3, depth=17, n_channels=64).cuda()
        return model
    elif arch == "cifar_dncnn_wide":
        model = DnCNN(image_channels=3, depth=17, n_channels=128).cuda()
        return model
    elif arch == 'memnet':
        model = MemNet(in_channels=3, channels=64, num_memblock=3, num_resblock=6).cuda()
        return model
    elif arch == "imagenet_dncnn":
        model = torch.nn.DataParallel(DnCNN(image_channels=3, depth=17, n_channels=64)).cuda()
        cudnn.benchmark = True
        return model
    elif arch == 'imagenet_memnet':
        model = torch.nn.DataParallel(MemNet(in_channels=3, channels=64, num_memblock=3, num_resblock=6)).cuda()
        cudnn.benchmark = True
        return model
    else:
        raise Exception('Unknown architecture.')

    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
