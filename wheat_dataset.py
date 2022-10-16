from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
import mmcv
import pandas as pd
import numpy as np

@DATASETS.register_module()
class WheatDataset(CustomDataset):
    CLASSES = ('Head',)
    PALETTE = (225,100,196)
    def load_annotations(self, ann_file):
        print(f'loading annotations for WheatDataset from {ann_file}...')
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(ann_file)
    
        data_infos = []

        marking = pd.read_csv('data/annotations.csv')

        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.jpg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.jpg', width=width, height=height)

            if 'train' in self.img_prefix:
                boxes = np.zeros((0, 4), dtype=np.float32)
                if image_id in marking['image_id'].values:
                    # load annotations     
                    boxes = marking[marking['image_id'] == image_id][['x', 'y', 'w', 'h']].to_numpy(dtype=np.float32)
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

                data_anno = dict(
                    bboxes=boxes,
                    labels=np.zeros(len(boxes), dtype=np.int64)
                    )
                data_info.update(ann=data_anno)

            data_infos.append(data_info)

        return data_infos