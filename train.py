# %%
import os
import wandb
wandb.login(key="52be99a40710a38857e86cb163238de4e437a074")

# %% [markdown]
# ### 1. Copy file to train and val folder

# %% [markdown]
# ### 2. Load dataset

# %%
import torchvision
import os

base_model = "vietlethe/bkad-deformable-detr"

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "annotations.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


# %%
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained(base_model)

train_dataset = CocoDetection(img_folder='dataset/train', processor=processor)
val_dataset = CocoDetection(img_folder='dataset/val', processor=processor, train=False)

# %%
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

# %%
import numpy as np
import os
from PIL import Image, ImageDraw

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

# %%
from torch.utils.data import DataLoader

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, num_workers=13, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, num_workers=13, batch_size=4)
batch = next(iter(train_dataloader))

# %%
batch.keys()

# %%
pixel_values, target = train_dataset[0]
print(pixel_values.shape)
print(target)

# %% [markdown]
# ## 3.Train the model

# %%
import pytorch_lightning as pl
from transformers import DeformableDetrForObjectDetection
import torch

base_model = "vietlethe/bkad-deformable-detr"

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DeformableDetrForObjectDetection.from_pretrained(base_model,
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader

# %%
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

# %%
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

# %%
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import pytorch_lightning as pl
from typing import Any, Dict, Optional
import torch
import logging
import sys
import shutil

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set up the root lightning logger
lightning_logger = logging.getLogger("lightning.pytorch")
lightning_logger.setLevel(logging.INFO)

# Add both file and console handlers
file_handler = logging.FileHandler("training.log")
file_handler.setFormatter(formatter)
lightning_logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
lightning_logger.addHandler(console_handler)

class TrainingCallback(pl.Callback):
    def __init__(
        self,
        monitor_metric: str = "val_loss",
        patience: int = 3,
        min_delta: float = 0.001,
        save_path: Optional[str] = None,
        logging_interval: int = 1,
        evaluate_mAP_interval: int = 2
    ):
        super().__init__()
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.logging_interval = logging_interval
        self.evaluate_mAP_interval = evaluate_mAP_interval
        
        self.best_score = float('inf')
        self.counter = 0
        # Use the existing lightning logger instead of creating a new one
        self.logger = logging.getLogger("lightning.pytorch")
        
        # Test the logger
        self.logger.info("Callback initialized successfully")

        self.logger.info("logger init")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training begins."""
        self.logger.info(f"Starting training with {trainer.max_epochs} epochs")
        
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when a training epoch begins."""
        self.logger.info(f"Starting epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
        
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int
    ) -> None:
        """Called when a training batch ends."""
        if batch_idx % self.logging_interval == 0:
            self.logger.info(
                f"Epoch {trainer.current_epoch + 1}, "
                f"Batch {batch_idx}, "
                f"Loss: {outputs['loss'].item():.4f}"
            )

    def save_hf_model(self, name):
        self.logger.info(f"save {name} model")
        try:
          model.model.push_to_hub(name)
          processor.push_to_hub(name)
        except:
            self.logger.error("cannot push model to hub!")

        if os.path.exists(name):
          shutil.rmtree(name)

        model.model.save_pretrained(name)
        processor.save_pretrained(name)

        self.logger.info(f"save {name} model completed!")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when validation ends. Handles early stopping and model saving logic.
        """
        # Get current validation metric
        current_score = trainer.callback_metrics.get(self.monitor_metric)
        
        if current_score is None:
            self.logger.warning(f"Metric {self.monitor_metric} not found in callback_metrics")
            return
        
        last_path = "vietlethe/bkad-deformable-detr_last"
        best_path = "vietlethe/bkad-deformable-detr_best"

        self.save_hf_model(last_path)

        # Check if score improved
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0

            self.logger.info(f"Saving best model with {self.monitor_metric}: {current_score:.4f}")
            self.save_hf_model(best_path)
                # torch.save(pl_module.state_dict(), self.save_path)
        else:
            self.counter += 1
            
        # Early stopping check
        if self.counter >= self.patience:
            self.logger.info(
                f"Early stopping triggered: no improvement in {self.monitor_metric} "
                f"for {self.patience} epochs"
            )
            trainer.should_stop = True

        if trainer.current_epoch + 1 % self.evaluate_mAP_interval == 0:
            pass
          #  evaluate_mAP()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends."""
        self.logger.info(
            f"Training completed. Best {self.monitor_metric}: {self.best_score:.4f}"
        )

wandb_logger = WandbLogger(log_model="all")

callback = TrainingCallback(
    monitor_metric="validation_cardinality_error",
    patience=10,
    save_path="best_model.pt",
    logging_interval=50
)

trainer = Trainer(
    max_steps=-1,
    max_epochs=10,
    gradient_clip_val=0.1,
    accelerator="gpu",
    devices=[0],
    callbacks=[callback],
    logger=wandb_logger)

trainer.fit(model)

# %%
model.model.push_to_hub("vietlethe/bkad-deformable-detr_last")
processor.push_to_hub("vietlethe/bkad-deformable-detr_last")