import os
import os.path as osp
import random
import multiprocessing

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES, build_dataset

class ImgLoader:
    def __init__(self,
                img_prefix,
                shuffle=True):
                
        assert os.path.exists(img_prefix), '%s not exists'%img_prefix
        self.img_prefix = img_prefix
        self.load_img = build_from_cfg(dict(type='LoadImageFromFile'), PIPELINES)
        self.shuffle = shuffle
        self.glob_idx = multiprocessing.Value('i', 0)

        self.names = []
        for root, _, files in os.walk(img_prefix):
            for f in files:
                if f.endswith('jpg') or f.endswith('png'):
                    self.names.append(osp.join(root, f).replace(img_prefix, ''))
        self.length = len(self.names)

        if self.shuffle:
            random.shuffle(self.names)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.glob_idx.get_lock():
            img = self.get_img_by_name(self.names[self.glob_idx.value])
            self.glob_idx.value += 1
            if self.glob_idx.value % self.length == 0:
                self.glob_idx.value = 0
                if self.shuffle:
                    random.shuffle(self.names)
        return img


    def get_img_by_name(self, filename):
        result = dict(
            img_prefix=self.img_prefix,
            img_info={'filename':filename}
        )
        result = self.load_img(result)
        return result


@DATASETS.register_module()
class LCPDataset:
    '''
    Load UI, UI_H and DFUI_H at the same time

    Args:
        dataset (dict): config of coco dataset for UI
        pipeline (list[dict]): Processing pipeline.
        img_dfui_prefix (str, optional): DFUI_H image prefiex
        mask_thr (float, optional): mask threshold
        test_mode (bool, optional): If set True, annotation will not be loaded.
    '''
    def __init__(self,
                 dataset,
                 pipeline,
                 img_df_prefix=None,
                 test_mode=False):
        self.dataset = build_dataset(dataset)
        self.dataset.test_mode = test_mode
        self.CLASSES = self.dataset.CLASSES
        self.flag = self.dataset.flag
        self.PALETTE = getattr(dataset, 'PALETTE', None)

        self.img_df_loader = ImgLoader(img_df_prefix) if img_df_prefix is not None else None

        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        result = self.dataset[idx]
        result['img_fields'] = ['img']

        # load detection favouring image
        if self.img_df_loader is not None:
            result['img_df'] = self.img_df_loader[idx]['img']
            result['img_fields'].insert(0, 'img_df')
            
        result = self.pipeline(result)
        return result
   
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        return self.dataset.evaluate(
            results,
            metric,
            logger,
            jsonfile_prefix,
            classwise,
            proposal_nums,
            iou_thrs,
            metric_items
        )
