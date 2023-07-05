'''Fine tuning ViT on CIFAR-100 using PyTorch Lightning'''

from datasets import load_dataset
from transformers import ViTImageProcessor
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from transformers import ViTForImageClassification, AdamW
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import subprocess
import os


'''
load portion of cifar 100
'''
train_ds, test_ds = load_dataset('cifar100', split=['train[:]', 'test[:]'])


'''
training & validation splits
'''
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

# print(train_ds.shape)
# print(train_ds.features)


'''
mapping label id to label name
'''
id2finelabel = {id: label for id, label in enumerate(train_ds.features['fine_label'].names)}
id2coarselabel = {id: label for id, label in enumerate(train_ds.features['coarse_label'].names)}
finelabel2id = {label: id for id, label in id2finelabel.items()}
coarselabel2id = {label: id for id, label in id2coarselabel.items()}


'''
pre-processing the data - data augmentation on the fly using set_transform method
'''
# image processing
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size['height']

# data augmentation using torchvision's transforms module
normalize = Normalize(mean=image_mean, std=image_std)
_train_tranforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

_val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)


def train_transforms(examples):
    examples['pixel_values'] = [_train_tranforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

# set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)


'''
create PyTorch DataLoaders
'''
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    fine_labels = torch.tensor([example['fine_label'] for example in examples])
    coarse_labels = torch.tensor([example['coarse_label'] for example in examples])
    return {"pixel_values": pixel_values, "fine_labels": fine_labels, "coarse_labels": coarse_labels}

train_batch_size = 32
eval_batch_size = 32

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

batch = next(iter(train_dataloader))

assert batch['pixel_values'].shape == (train_batch_size, 3, 224, 224)
assert batch['fine_labels'].shape == (train_batch_size,)
assert batch['coarse_labels'].shape == (train_batch_size,)


''' define model - using LightningModule
    model uses linear layer on top of a pre-trained ViTModel
    linear layer on top of the last hidden state of [CLS] token - good representation of the entire image
'''
class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=20):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=100,
                                                              id2label=id2finelabel,
                                                              label2id=finelabel2id)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
    
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        fine_labels = batch['fine_labels']
        coarse_labels = batch['coarse_labels']
        logits = self(pixel_values)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, fine_labels)
        predictions = logits.argmax(-1)
        correct = (predictions == fine_labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        return loss, accuracy
      

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)    

        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        return loss
    

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)
    

    def train_dataloader(self):
        return train_dataloader
    

    def val_dataloader(self):
        return val_dataloader
    

    def test_dataloader(self):
        return test_dataloader
    

''' 
start tensorboard
'''
writer = SummaryWriter(log_dir='lightning_logs')
subprocess.Popen(['tensorboard', '--logdir', 'lightning_logs'])


''' initialize and train model 
    callback - early stopping : stop training once validation loss stops improving 3 times in a row
'''
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)

model = ViTLightningModule()
trainer = Trainer(accelerator='gpu', callbacks=[EarlyStopping(monitor='validation_loss')])
trainer.fit(model)
trainer.test()
my_path = os.path.dirname(__file__)
torch.save(model, f"{my_path}/vit_cifar100.pth")
