import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray to a torch.FloatTensor of shape (C x H x W)
    # Other transformations can be added here, e.g.,
    # transforms.Resize((256, 256)),  # Resize to (256x256)
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Example normalization if needed
])

class RefDataset(Dataset):
    def __init__(self, parquet_file, seed = 42, transform=transform):
        super().__init__()
        if isinstance(parquet_file, pd.DataFrame):
            self.df = parquet_file
        else:
            # 如果 data 不是 DataFrame，则假设是一个文件路径并读取
            self.df = pd.read_parquet(parquet_file)
        self.transform = transform
        self.rng = random.Random(seed)
    
    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, index):
        row = self.df.iloc[index]

        file_name = row['file_name']
        image_id = file_name.split('_')[2] + '.jpg'
        image_path = os.path.join('/mnt/pfs-mc0p4k/nlu/team/yuhaofu/model/Visual-CoT/playground/data/coco/train2014', image_id)
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path {image_path} is invalid")
        
        image = Image.open(image_path).convert("RGB")

        ann = row['answer']
        random_ann = self.rng.choice(ann)

        w, h = image.size
        bbox = row['bbox']
        norm_bbox = self.normalize_bbox(bbox, w, h)
         # 将每个边界框坐标截取到小数点后三位
        norm_bbox = [round(coord, 3) for coord in norm_bbox]

        # 转换为一个 PyTorch 张量
        norm_bbox_tensor = torch.tensor(norm_bbox, dtype=torch.float32)

        # Apply the transform to the image
        if self.transform:
            image = self.transform(image)


        return random_ann, image, norm_bbox_tensor, image_path


    def normalize_bbox(self, bbox, w, h):
        # 计算左上和右下坐标
        x, y, width, height = bbox
        xmin = x
        ymin = y
        xmax = x + width
        ymax = y + height

        # 归一化
        xmin_norm = xmin / w
        ymin_norm = ymin / h
        xmax_norm = xmax / w
        ymax_norm = ymax / h

        return [xmin_norm, ymin_norm, xmax_norm, ymax_norm]
    



class IoUEvaluator:
    def __init__(self, iou_threshold=0.5):
        """
        初始化 IoU 评估器
        :param iou_threshold: IoU的阈值，用于判断预测结果是否正确
        """
        self.iou_threshold = iou_threshold
        self.total_predictions = 0
        self.correct_predictions = 0

    def calculate_iou(self, pred_bbox, true_bbox):
        """
        计算预测边界框与真实边界框的IoU
        :param pred_bbox: [xmin, ymin, xmax, ymax]，预测边界框
        :param true_bbox: [xmin, ymin, xmax, ymax]，真实边界框
        :return: IoU值
        """
        # 求交集
        ixmin = max(pred_bbox[0], true_bbox[0])
        iymin = max(pred_bbox[1], true_bbox[1])
        ixmax = min(pred_bbox[2], true_bbox[2])
        iymax = min(pred_bbox[3], true_bbox[3])
        
        iw = max(0, ixmax - ixmin)
        ih = max(0, iymax - iymin)
        intersection = iw * ih

        # 求并集
        pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
        true_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
        union = pred_area + true_area - intersection
        
        # 计算IoU
        iou = intersection / union if union > 0 else 0
        return iou

    def evaluate(self, pred_bbox, true_bbox):
        """
        评估单个预测结果，并更新统计信息
        :param pred_bbox: [xmin, ymin, xmax, ymax]，预测边界框
        :param true_bbox: [xmin, ymin, xmax, ymax]，真实边界框
        :return: bool - 是否预测正确
        """
        self.total_predictions += 1
        iou = self.calculate_iou(pred_bbox, true_bbox)
        if iou >= self.iou_threshold:
            self.correct_predictions += 1
            return True
        return False

    def accuracy(self):
        """
        计算当前所有评估的准确率
        :return: 准确率
        """
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions