from collections import OrderedDict, defaultdict
import glob
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

from PIL import Image
#import cv2

import albumentations as A

import gzip

from torch.nn.functional import relu, avg_pool2d
import numpy as np
import pickle
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from tqdm import tqdm


class PadSquare:
    def __call__(self, img):
        w, h = img.size
        diff = abs(w - h)
        p1 = diff // 2
        p2 = p1
        if diff % 2 == 1:
            p2 += 1
        if w > h:
            return transforms.functional.pad(img, (0, p1, 0, p2))
        else:
            return transforms.functional.pad(img, (p1, 0, p2, 0))

    def __repr__(self):
        return self.__class__.__name__

def transform_image_ifcc(split = 'none'):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rotate = transforms.RandomApply([transforms.RandomRotation(10.0, expand=True)], p=0.5)
    color = transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5)
    if split == 'train':
        augs = [rotate, color]
        augs += [PadSquare(),transforms.Resize(224)]
    else:
        augs =[]
    return (
        transforms.Compose(
            #[PadSquare(),
            # transforms.Resize(224)] +
            augs + [transforms.ToTensor(),
                    norm]
        )
    )


class IuxrayMultiImageDataset(Dataset):
    def __init__(self, X, y, uid=None): # type = 'train' or 'test' or 'val'
        self.X = X
        self.y = y
        self.uid=uid

        self.transform = transform_image_ifcc(type)

    def __len__(self):
        return(len(self.y))

    def __getitem__(self, idx):
        image_1, image_2 = self.X[idx]
        label = self.y[idx]
        image_1 = self.transform(image_1)
        image_2 = self.transform(image_2)

        if self.uid !=None: #test dataset
            uid = self.uid[idx]
            return (image_1, image_2, label, uid)
        else: # train, val dataset
            return (image_1, image_2, label)
            

def augment_dataset(X_train, y_train):
    train_hist, _ = np.histogram(y_train, bins=[0,1,2,3])  

    train_neg_num = train_hist[0]
    train_pos_num = train_hist[1]

    aug_elastic = A.ElasticTransform(p=1, alpha=200, sigma=200 * 0.08, alpha_affine=200 * 0.08)
    aug_grid  = A.GridDistortion(p=1)

    aug_X_train = X_train.copy()
    aug_y_train = y_train.copy()
    while train_neg_num > train_pos_num:
        for X, label in zip(X_train, y_train):
            image_1, image_2 = X
            if label == 1: # positive
                image_1 = np.array(image_1)
                image_2 = np.array(image_2)

                elastic_image_1 = aug_elastic(image=image_1)['image']
                elastic_image_2 = aug_elastic(image=image_2)['image']
                grid_image_1 = aug_grid(image=image_1)['image']
                grid_image_2 = aug_grid(image=image_2)['image']

                elastic_image_1 = Image.fromarray(elastic_image_1)
                elastic_image_2 = Image.fromarray(elastic_image_2)
                grid_image_1 = Image.fromarray(grid_image_1)
                grid_image_2 = Image.fromarray(grid_image_2)

                #add 2 augmented data
                aug_X_train.append((elastic_image_1, elastic_image_2))
                aug_y_train.append(label) 
                aug_X_train.append((grid_image_1, grid_image_2))
                aug_y_train.append(label)

                train_pos_num+=2

            if train_neg_num <= train_pos_num:
                break

    return aug_X_train, aug_y_train


class ClassifierDenseNet121(nn.Module):
    def __init__(self):
        super(VisualExtractorDenseNet121, self).__init__()
        self.visual_extractor = 'densenet121'
        self.cached_file = "/cluster/work/medinfmk/ARGON/containers/models/chexpert_auc14.dict.gz"
        if os.path.exists(self.cached_file):
            self.pretrained = False
        else:
            self.pretrained = True
        #weights='DEFAULT'

        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)

        #model.load_state_dict(torch.load("/cluster/home/sankim/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth"))

        if not self.pretrained:
            print("loading densnet parameters from ", self.cached_file)
            #logger.info("loading densnet parameters from {}".format(self.cached_file))
            with gzip.open(self.cached_file) as f:
                state_dict = torch.load(f, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=False)

        # modules = list(model.children())[:-2]
        #self.model = model
        self.model_1 = nn.Sequential(*list(model.features.children()))
        self.model_2 = deepcopy(self.model_1)
        self.avg_fnt = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        #self.fine_tune(fine_tune=True)
        self.flatten = nn.Flatten()
        self.fc  = nn.Linear(2048, 256)
        self.fc1   = nn.Linear(256, 32)
        self.fc2  = nn.Linear(32, 2)


    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.model.parameters():
            p.requires_grad = fine_tune


    def forward(self, image_1, image_2):
        # CNN features
        patch_feat_1 = self.model_1(image_1)
        avg_feat_1 = self.flatten(self.avg_fnt(patch_feat_1)) #torch.Size([batch_size, 1024])

        patch_feat_2 = self.model_2(image_2)
        avg_feat_2 = self.flatten(self.avg_fnt(patch_feat_2)) #torch.Size([batch_size, 1024])

        fc_feat = torch.cat((avg_feat_1, avg_feat_2), dim=1)

        y = relu(self.fc(fc_feat))
        y = relu(self.fc1(y))
        y = self.fc2(y)

        return y

class Comparison_Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VisualExtractorDenseNet121()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images1, images2, labels = batch
        outputs = self.model(images1, images2)
        loss = nn.functional.cross_entropy(outputs, labels)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images1, images2, labels = batch
        outputs = self.model(images1, images2)
        loss = nn.functional.cross_entropy(outputs, labels)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        total_loss = 0
        for batch_result in outputs:
            total_loss += batch_result["loss"]
        total_mean = total_loss/len(outputs)
        self.log("total_val_loss", total_mean )

    def test_step(self, batch, batch_idx):
        output_dict = {}
        images1, images2, labels, uids = batch
        outputs = self.model(images1, images2)
        outputs = outputs.softmax(dim=1)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        correct_neg = ((predicted == labels) & (labels == 0)).sum().item()
        correct_pos = ((predicted == labels) & (labels == 1)).sum().item()

        total = labels.size(0)
        total_neg = (labels == 0).sum().item()
        total_pos = (labels == 1).sum().item()

        output_dict["correct"] = correct
        output_dict["correct_neg"] = correct_neg
        output_dict["correct_pos"] = correct_pos
        output_dict["total"] = total
        output_dict["total_neg"] = total_neg
        output_dict["total_pos"] = total_pos

        output_dict["probs_neg"] = outputs[labels == 0, 0].tolist()
        output_dict["probs_pos"] = outputs[labels == 1, 1].tolist()
        print("#########################")
        print(uids)
        print(predicted)
        print(labels)

        return output_dict

    def test_epoch_end(self, outputs):
        correct = 0
        correct_neg = 0
        correct_pos = 0
        correct_unc = 0   
        total = 0
        total_neg = 0
        total_pos = 0
        total_unc = 0 

        total_probs_neg = []
        total_probs_pos = []

        for batch_result in outputs:
            correct += batch_result["correct"]
            correct_neg += batch_result["correct_neg"]
            correct_pos += batch_result["correct_pos"]
            total += batch_result["total"]
            total_neg += batch_result["total_neg"]
            total_pos += batch_result["total_pos"]

            total_probs_neg = total_probs_neg + batch_result["probs_neg"]
            total_probs_pos = total_probs_pos + batch_result["probs_pos"]         

        acc = round(100 * correct / total, 2)
        acc_neg = round(100 * correct_neg / total_neg, 2)
        acc_pos = round(100 * correct_pos / total_pos, 2)

        self.log("test_acc", acc) 
        self.log("test_acc_neg", acc_neg) 
        self.log("test_acc_pos", acc_pos) 

        print(f'Total_accuracy : {acc} %, Negative {acc_neg} %, Positive {acc_pos} %')   
        print(len(total_probs_neg), np.histogram(total_probs_neg, bins=10, range=[0,1])[0])
        print(len(total_probs_pos), np.histogram(total_probs_pos, bins=10, range=[0,1])[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer, patience=10, mode='min', factor=0.2 , verbose=True), 
                            "monitor": "total_val_loss"}
        return [optimizer],[lr_schedulers]


if __name__ == '__main__':  
    image_dir = "/cluster/work/medinfmk/ARGON/openi/images/"
    file_names = glob.glob("/cluster/work/medinfmk/ARGON/openi/images/*.png")
    #print(len(file_names))

    torch.manual_seed(9223)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(9223)

    reports = pd.read_csv('labeled_iu_reports.csv', 
                                        header=0,
                                      names=["uid","Report","Keyword","Label"])
    uids = reports["uid"].tolist()
    labels = reports["Label"].tolist()
    label_dict = {}
    for i, uid in enumerate(uids):
        label = labels[i]
        label_dict[str(uid)] = label

    projection_dict = defaultdict(list)
    for file in file_names:
        dir_split = file.split('/')
        file_split = dir_split[-1].split('_')
        key = file_split[0][3:]
        projection_dict[key].append(dir_split[-1])

    short_images = []
    for k, v in projection_dict.items():
        if len(v)==1:
            short_images.append(k)
    for k in short_images:
        del projection_dict[k]

    trans = transforms.Compose(
            [PadSquare(),
             transforms.Resize(224)])
    key_list = list(projection_dict.keys())
    data_dict = {"uid": [], "images": [], "labels": []}
    for idx, key in enumerate(key_list):
        key = key_list[idx]
        file_name_lst = projection_dict[key]

        image_1 = Image.open(os.path.join(image_dir, file_name_lst[0])).convert('RGB')
        image_2 = Image.open(os.path.join(image_dir, file_name_lst[1])).convert('RGB')

        image_1 = trans(image_1)
        image_2 = trans(image_2)

        label = label_dict[key]

        data_dict["uid"].append(key)
        data_dict["images"].append((image_1, image_2))
        data_dict["labels"].append(label)

    uid_train, uid_test, X_train, X_test, y_train, y_test = train_test_split(data_dict["uid"], data_dict["images"], data_dict["labels"], test_size=0.2, 
                                                        random_state=42, stratify = data_dict["labels"])
   
    train_hist, _ = np.histogram(y_train, bins=[0,1,2,3])
    test_hist, _ = np.histogram(y_test, bins=[0,1,2,3])
    print("original train_histogram", train_hist)
    print("original test_histogram", test_hist)

    X_train, y_train = augment_dataset(X_train, y_train)
    train_hist, _ = np.histogram(y_train, bins=[0,1,2,3])
    print("augmented train_histogram", train_hist)    

    """
    TRANSFORM_TRAIN=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ])
    
    #Random Crop and Flip train images
    for i, images in enumerate(X_train):
        image1, image2 = images
        X_train[i] = (TRANSFORM_TRAIN(image1), TRANSFORM_TRAIN(image2))
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, 
                                                        random_state=42, stratify = y_train)  

    train_hist, _ = np.histogram(y_train, bins=[0,1,2,3])
    val_hist, _ = np.histogram(y_val, bins=[0,1,2,3])
    print("split_train_histogram", train_hist)
    print("split_validation_histogram", val_hist)


    train_dataset = IuxrayMultiImageDataset(X_train, y_train)
    val_dataset = IuxrayMultiImageDataset(X_val, y_val)
    test_dataset = IuxrayMultiImageDataset(X_test, y_test, uid_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    checkpoint_callback = ModelCheckpoint(
            monitor="total_val_loss",
            dirpath = "./records",
            filename="model_best4",
            mode="min",
        )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    classifier = Comparison_Classifier()

    trainer = pl.Trainer(
            devices=1, accelerator="auto",
            max_epochs=100, deterministic=True,
            callbacks=[EarlyStopping(monitor="total_val_loss", mode="min", patience=50, verbose=True), 
            checkpoint_callback, lr_monitor]
        )
    trainer.fit(classifier, train_loader, val_loader)
    trainer.test(classifier, dataloaders=test_loader)
    #trainer.test(model = classifier, ckpt_path="./records/model_best.ckpt", dataloaders=test_loader)
