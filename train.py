import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from model import Yolov1
from tqdm import tqdm
from dataset import VOCDataset
from loss import YoloLoss
from utils import \
    (non_max_suppression,
     mean_average_precision,
     intersection_over_union,
     cellboxes_to_boxes,
     get_bboxes,
     plot_image,
     save_checkpoint,
     load_checkpoint)

seed = 42
torch.manual_seed(seed)

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "yolov1.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
CSV_FILE = "./data/100examples.csv"
SPLIT_SIZE = 7
IMG_SIZE = 448
NUM_CLASSES = 20
CONF_THRESH = 0.5
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.5


class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = CustomCompose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),])

def train_step(train_dataloader, model, optimizer, loss_fn):
    mean_loss = []
    loop = tqdm(train_dataloader, leave = True)
    for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss = loss.item())
    print(f"Mean Loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size = SPLIT_SIZE, num_boxes = 2, num_classes = NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(csv_file = CSV_FILE, img_dir = IMG_DIR, label_dir = LABEL_DIR, transform = transform)
    test_dataset = VOCDataset(csv_file = CSV_FILE, img_dir = IMG_DIR, label_dir = LABEL_DIR, transform = transform)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True, drop_last = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True, drop_last = True)

    for epoch in tqdm(range(EPOCHS)):
        pred_boxes, target_boxes = get_bboxes(train_dataloader, model, iou_threshold = NMS_IOU_THRESH, threshold = CONF_THRESH)
        train_step(train_dataloader, model, optimizer, loss_fn)
        mean_average_precision = mean_average_precision(pred_boxes, target_boxes, iou_threshold = MAP_IOU_THRESH, box_format = "midpoint")
        print(f"Train mAP: {mean_average_precision}")



if __name__ == "__main__":
    main()
