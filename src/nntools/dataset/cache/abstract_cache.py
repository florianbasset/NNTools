from abc import abstractmethod
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np
import tqdm
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from nntools.dataset.abstract_image_dataset import AbstractImageDataset


class AbstractCache:
    def __init__(self, dataset):
        self.d: AbstractImageDataset = dataset
        self.is_initialized = False
        self.shms = []
        self._num_workers = 12
        self._batch_size = 32
        self.is_item_cached = None

    def init_shared_items_tracking(self):
        cached_items = np.zeros(self.nb_samples, dtype=bool)
        try:
            shm = shared_memory.SharedMemory(
                name=f"nntools_{self.id}_is_item_cached", size=cached_items.nbytes, create=True
            )
            buffer = np.frombuffer(buffer=shm.buf, dtype=bool)
            buffer[:] = 0
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=f"nntools_{self.id}_is_item_cached", create=False)

        self.shms.append(shm)
        self.is_item_cached = np.frombuffer(buffer=shm.buf, dtype=bool)

    def init_non_shared_items_tracking(self):
        self.is_item_cached = np.zeros(self.nb_samples, dtype=bool)

    @property
    def id(self):
        return self.d.id

    @abstractmethod
    def remap(self, old_key: str, new_key: str):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def init_cache(self):
        pass

    @property
    def nb_samples(self):
        return self.d.real_length

    def auto_cache(self, method="thread"):
        self.init_cache()
        dataloader = DataLoader(
            self.d, num_workers=self._num_workers, batch_size=self._batch_size, pin_memory=False, shuffle=False
        )
        for i in tqdm.tqdm(dataloader, total=len(dataloader)):
            pass
