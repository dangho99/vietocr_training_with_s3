import torch
import numpy as np
from torch.utils.data import IterableDataset

from src.data.s3 import S3IterableDataset
from src.utils.const import DataConst

class OCRDataset(IterableDataset):
    def __init__(self, path_lists, shuffle_path=False, transform=None):
        self.s3_iter_dataset = S3IterableDataset(path_lists=path_lists,
                                                 shuffle_path=shuffle_path)
        self.transform = transform

    def data_generator(self):
        try:
            while True:
                label, image = next(self.s3_iter_dataset_iterator)
                if self.transform is not None:
                    image = self.transform(image)
                    image = self.transform.process_image(
                        image, DataConst.image_height, DataConst.image_min_width, DataConst.image_max_width
                    )
                    label = DataConst.vocab.encode(label)
                yield image, label

        except StopIteration:
            raise StopIteration

    def set_epoch(self, epoch):
        self.s3_iter_dataset.set_epoch(epoch)

    def __iter__(self):
        self.s3_iter_dataset_iterator = iter(self.s3_iter_dataset)
        return self.data_generator()


class Collator(object):
    def __init__(self, is_masked) -> None:
        self.is_masked = is_masked
        
    def __call__(self, batch):
        img = []
        target_weights = []
        tgt_input = []
        max_label_len = max(len(sample[1]) for sample in batch)
        for sample in batch:
            img.append(sample[0])
            label = sample[1]
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
        if self.is_masked:
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
    
