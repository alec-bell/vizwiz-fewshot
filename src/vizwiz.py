import json
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ObjectDetectionDataset(Dataset):

    def __init__(self, root, annFile, transform=lambda identity: identity, fold=0, set_type='base', shots=1):
        self.root = Path(root)
        self.transform = transform

        with open(str(annFile)) as file_obj:
            self.vizwiz_data = json.load(file_obj)
        
        self.categories = dict()
        for cat in self.vizwiz_data['categories']:
            self.categories[cat['id']] = cat

        self.image_id_to_annos = {}
        for anno in self.vizwiz_data['annotations']:
            image_id = anno['image_id']
            try:
                cat = self.categories[anno['category_id']]
            except:
                print(anno)
            if (set_type == 'base' and cat['fold'] != fold) or \
               (set_type == 'support' and len(self.image_id_to_annos[image_id]) < shots and cat['fold'] == fold) or \
               (set_type == 'query' and cat['fold'] == fold):
                if image_id not in self.image_id_to_annos:
                    self.image_id_to_annos[image_id] = []
                self.image_id_to_annos[image_id] += [anno]

    def __len__(self):
        return len(self.vizwiz_data['images'])

    def __getitem__(self, index):
        image_data = self.vizwiz_data['images'][index]
        image_id = image_data['id']
        image_path = self.root/image_data['file_name']
        image = Image.open(image_path)

        annos = self.image_id_to_annos[image_id]
        anno_data = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': []
        }
        for anno in annos:
            vizwiz_bbox = anno['bbox']
            left = vizwiz_bbox[0]
            top = vizwiz_bbox[1]
            right = vizwiz_bbox[0] + vizwiz_bbox[2]
            bottom = vizwiz_bbox[1] + vizwiz_bbox[3]
            area = vizwiz_bbox[2] * vizwiz_bbox[3]
            anno_data['boxes'].append([left, top, right, bottom])
            anno_data['labels'].append(anno['category_id'])
            anno_data['area'].append(area)

        target = {
            'boxes': torch.as_tensor(anno_data['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(anno_data['labels'], dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'area': torch.as_tensor(anno_data['area'], dtype=torch.float32)
        }

        return self.transform(image), target
