""" S3 Dataset class ref: https://github.com/aws/amazon-s3-plugin-for-pytorch/blob/master/awsio/python/lib/io/s3/s3dataset.py#L154"""

import random
import torch
from itertools import chain
from src.data.connector import MinioConnector
from src.utils.const import EnvConst
from torch.utils.data import IterableDataset
import torch.distributed as dist

class S3IterableDataset(IterableDataset):
    def __init__(self, path_lists: list, shuffle_path=False):
        self.path_lists = path_lists
        self.shuffle_path = shuffle_path
        self.epoch = 0
        self.dist = dist.is_initialized() if dist.is_available() else False
        if self.dist:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

    @property
    def shuffled_list(self):
        if self.shuffle_path:
            random.seed(self.epoch)
            return random.sample(self.path_lists, len(self.path_lists))
        else:
            return self.path_lists
        
    def download_data(self, element:list):
        path = element[0]
        label = element[1]
        data = self.minio_client.read_image(EnvConst.minio_bucket_name, path)
        for i in data:
            yield label, i

    def get_stream(self, path_lists):
        return chain.from_iterable(map(self.download_data, path_lists))
    
    def worker_dist(self, urls):
        if self.dist:
            total_size = len(urls)
            urls = urls[self.rank:total_size:self.world_size]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            wid = worker_info.id
            num_workers = worker_info.num_workers
            length = len(urls)
            return urls[wid:length:num_workers]
        else:
            return urls

    def __iter__(self):
        self.minio_client = MinioConnector()
        urls = self.worker_dist(self.shuffled_list)
        return self.get_stream(urls)

    def __len__(self):
        return len(self.path_lists)

    def set_epoch(self, epoch):
        self.epoch = epoch
