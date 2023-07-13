import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from common.img_utils import batch_tensor_to_images
from data.data_utils import collate_ground_truth_boxes
from data.voc_dataset import VOCDataset, VOC_CLASSES

from ssd.anchor import AnchorGenerator
from ssd.box_utils import point_form
from ssd.encoder import TargetEncoder
from ssd.loss import MultiBoxLoss
from ssd.model import SingleShotDetector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


def data_loader():
    generator = AnchorGenerator()
    anchor_boxes_by_layer = generator.anchor_boxes
    priors_boxes = []
    for layer in anchor_boxes_by_layer:
        priors_boxes.append(anchor_boxes_by_layer[layer].view(-1, 4))
    priors_boxes = torch.cat(priors_boxes, dim=0)
    priors_boxes = point_form(priors_boxes, clip=True)

    encoder = TargetEncoder(priors_boxes, iou_threshold=0.5)

    train_dataset = VOCDataset("train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                              collate_fn=collate_ground_truth_boxes)
    for batch_imgs, batch_gts in train_loader:
        batch_matched_gts, batch_matched_labels, batch_loc_targets, batch_cls_targets = encoder.encode_batch(batch_gts[:, :, :4], batch_gts[:, :, 4])
        print("Batch Matched Ground Truths", batch_matched_gts.shape)
        print("Batch Matched Labels", batch_matched_labels.shape)
        print("Batch Location Targets", batch_loc_targets.shape)
        print("Batch Classification Targets", batch_cls_targets.shape)

    val_dataset = VOCDataset("val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
                            collate_fn=collate_ground_truth_boxes)

    for x, y in val_loader:
        print(x.shape, y.shape)


def test_tensorboard():
    generator = AnchorGenerator()
    anchor_boxes_by_layer = generator.anchor_boxes
    prior_boxes = []
    for layer in anchor_boxes_by_layer:
        prior_boxes.append(anchor_boxes_by_layer[layer].view(-1, 4))
    prior_boxes = torch.cat(prior_boxes, dim=0)
    prior_boxes = point_form(prior_boxes, clip=True)
    prior_boxes = prior_boxes.to(device)
    encoder = TargetEncoder(prior_boxes, iou_threshold=0.5)

    dataset = VOCDataset("trainval")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=1,
                              shuffle=True,
                              collate_fn=collate_ground_truth_boxes)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=1,
                            shuffle=False,
                            collate_fn=collate_ground_truth_boxes)

    summary_writer = SummaryWriter(log_dir=f"logs/test_tensorboard")

    class_names = {i + 1: VOC_CLASSES[i] for i in range(len(VOC_CLASSES))}
    class_names[0] = "background"

    global_step = 0
    for batch_img, batch_gt in tqdm(val_loader):
        batch_img = batch_img.to(device)
        batch_gt = batch_gt.to(device)

        images = batch_tensor_to_images(batch_img, batch_gt[:, :, :4], batch_gt[:, :, 4], class_names)
        summary_writer.add_images("ground_truths", images,
                                  global_step=global_step, dataformats='NHWC')
        global_step += 1

if __name__ == "__main__":
    test_tensorboard()
