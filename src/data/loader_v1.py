from typing import Iterator
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


from src.data.augment import ImgAugTransformV2
from src.connector import mongo_manager
from src.connector.minio import MinioConnector
from utils.const import DataConst

class OCRLoader(Dataset, DataConst, MinioConnector):
    def __init__(self, env_cfg_path, train_cfg_path: str):
        MinioConnector.__init__(self, env_cfg_path)
        DataConst.__init__(self,train_cfg_path)

        self.transform = ImgAugTransformV2()
        self.cluster_indices = None  
        self.df = None
        self._build_data_cluster()
        
    def _init_cluster(self):
        self.cluster_indices = defaultdict(list)  

    def _get_data(self, is_query=True):
        if is_query == True:
            data_list = mongo_manager.query_by_key_value("to_be_reviewed","Skip")
            data_list = data_list[:self.n_sample]
            self.df = pd.DataFrame(data_list)
        else:    
            pass
        
    def _build_data_cluster(self) -> defaultdict:
        self._init_cluster()
        self._get_data(is_query=True)
        pbar = tqdm(range(self.n_sample), 
                desc=f'building cluster with {self.n_sample}', 
                ncols = 100, position=0, leave=True)
        for i in pbar:
            bucket_name = self.df["pre_annotation_label"][i][5]
            self.cluster_indices[bucket_name].append(i)
        return self.cluster_indices
    

    def read_data(self, idx):
        img_path = str(self.df["path_save"][idx])
        label = self.df["pre_annotation_label"][idx][5]
        print(f"Path: {type(self.bucket_image_name)} and {type(img_path)}")
        img = self.read_image(self.bucket_image_name, 
                                       img_path)
        img = self.transform(img)
        img_processed = self.transform.process_image(
            img, self.image_height, self.image_min_width, self.image_max_width
        )
        word = self.vocab.encode(label)
        return img_processed, word

    def __getitem__(self, idx):
        img_processed, word = self.read_data(idx)
        sample = {"img": img_processed, "word": word}
        return sample
    



class OCRSampler(Sampler):
    def __init__(self, data_source, batch_size, train_cfg_path: str, shuffle):
        self.train_cfg_path = train_cfg_path
        self.data_source = data_source
        self.shuffle = shuffle
        self.batch_size = batch_size
        
    
    def __iter__(self) -> Iterator:
        cluster_items = list(self.data_source.cluster_indices.items())
        batch = []
        idx_pointers = {key: 0 for key in self.data_source.cluster_indices.keys()} 

        while True:
            for key, indices in cluster_items:
                if idx_pointers[key] < len(indices):  
                    batch.append(indices[idx_pointers[key]])
                    idx_pointers[key] += 1 
                    
                    if len(batch) == self.batch_size:  
                        yield batch
                        batch = [] 

            if all(idx_pointers[key] >= len(indices) for key, indices in cluster_items):
                break
        if batch:
            yield batch


    def __len__(self):
        return len(self.data_source)

class Collator(object):
    def __init__(self) -> None:
        pass
        
    def __call__(self, batch):
        img = []
        target_weights = []
        tgt_input = []
        max_label_len = max(len(sample["word"]) for sample in batch)
        for sample in batch:
            img.append(sample["img"])
            label = sample["word"]
            label_len = len(label)

            tgt = np.concatenate(
                (label, np.zeros(max_label_len - label_len, dtype=np.int32))
            )
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(
                np.concatenate(
                    (
                        np.ones(one_mask_len, dtype=np.float32),
                        np.zeros(max_label_len - one_mask_len, dtype=np.float32),
                    )
                )
            )

        img = np.array(img, dtype=np.float32)

        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1] = 0

        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights) == 0

        rs = {
            "img": torch.FloatTensor(img),
            "tgt_input": torch.LongTensor(tgt_input),
            "tgt_output": torch.LongTensor(tgt_output),
            "tgt_padding_mask": torch.BoolTensor(tgt_padding_mask)
        }

        return rs
    

    
        
        
    
        
    