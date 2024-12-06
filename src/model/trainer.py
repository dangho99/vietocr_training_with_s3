
from torch.utils.data import DataLoader, RandomSampler

from src.data.connector import MongoDBManager
from src.utils.const import TrainConst, DataConst
#from data.loader_v1 import OCRLoader, OCRSampler, Collator
from src.data.loader_v2 import OCRDataset, Collator 
from src.data.augment import ImgAugTransformV2

class Trainer:
    def __init__(self):
        self._data = self.load_data()

    # Debugging here
    def train(self):
        data_iter = iter(self._data)
        for i in range(10):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self._data)
                batch = next(data_iter)
        pass

    
    def _get_test_data(self, n_sample):
        path_lists_pd = MongoDBManager().query_by_key_value("to_be_reviewed","Skip")
        path_lists_pd = path_lists_pd[:n_sample]
        path_lists = []
        for i in path_lists_pd:
            label = i["pre_annotation_label"][5]
            path = i["path_save"]
            path_lists.append([path,label])
        return path_lists   
    

    def load_data(self):
        path_lists = self._get_test_data(10000)
        dataset = OCRDataset(
            path_lists=path_lists, shuffle_path=True, transform=ImgAugTransformV2()
        )
        collate_fn = Collator(is_masked=True)
        loader = DataLoader(
            dataset=dataset,
            batch_size=TrainConst.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            **DataConst.dataloader
        )
        
        return loader
        pass