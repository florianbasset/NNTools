import logging
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

from nntools.dataset.cache.abstract_cache import AbstractCache
from nntools.utils.io import read_image, save_image
from nntools.utils.misc import is_image

class Metadata:
    def __init__(self, cache_folder, is_image):
        self.cache_folder = cache_folder
        self.is_image = is_image

class DiskCache(AbstractCache):
    def __init__(self, dataset, cache_dir:Path='') -> None:
        super().__init__(dataset)
        
        self.cache_folders = {}
        self.in_memory_items = []
        self.shms = []
        self.root_cache_folder = None
        self.needs_filling = False
        self.cache_dir = cache_dir
        
        
    def init_cache(self) -> None:
        if self.is_initialized:
            return
        self.root_cache_folder = self.get_cache_folder()
        self.root_cache_folder = self.root_cache_folder

        arrays = self.d.read_from_disk(0)  # Taking the first element
        arrays = self.d.precompose_data(arrays)
        self._cache_items = np.zeros(self.nb_samples, dtype=bool)

        for k, v in arrays.items():
            if not isinstance(v, np.ndarray):
                assert k in self.d.gts, f"Key {k} not found in dataset ground truths. \
                As it is not a numpy array, it can't be cached (for now)."
                self.in_memory_items.append(k)
            else:
                k_cache_folder = self.root_cache_folder / k
                metadata = Metadata(k_cache_folder, is_image(v))
                self.cache_folders[k] = metadata

        self._cache_items[:] = False
        for k, v in self.cache_folders.items():
            v.cache_folder.mkdir(parents=True, exist_ok=True)
            logging.info(f"Creating cache folder {self.root_cache_folder}.")

        self.needs_filling = self.check_if_filling_is_needed()
        self.is_initialized = True
        if not self.needs_filling:
            self._cache_items[:] = True

    def check_if_filling_is_needed(self):
        needed = []
        for k, v in self.cache_folders.items():
            v = v.cache_folder
            needed.append(not v.exists() or not (len(list(v.iterdir())) == self.nb_samples))
        return any(needed)

    def __getitem__(self, item):
        if not self._cache_items[item]:
            self._cache_items[item] = self.check_cache(item)
        if self._cache_items[item]:
            return self.get_cached_item(item)
        else:
            return self.cache_item(item)

    def get_cached_item(self, item):
        data = {k: self.d.gts[k][item] for k in self.in_memory_items}
        name = self.d.filename(item)
        for k, metadata in self.cache_folders.items():
            if metadata.is_image:
                data[k] = read_image(str((metadata.cache_folder / name).with_suffix(".jpg")), cv2.IMREAD_UNCHANGED)
            else:
                data[k] = np.load((metadata.cache_folder / name).with_suffix(".npy"))
        return data
    
    def check_cache(self, item):
        name = self.d.filename(item)
        for k, metadata in self.cache_folders.items():
            if metadata.is_image:
                if not (metadata.cache_folder / Path(name).with_suffix(".jpg")).exists():
                    return False
            else:
                if not (metadata.cache_folder / Path(name).with_suffix(".npy")).exists():
                    return False
        return True

    def cache_item(self, item):
        arrays = self.d.read_from_disk(item)
        arrays = self.d.precompose_data(arrays)
        for k, v in arrays.items():
            if k in self.in_memory_items:
                continue
            else:
                self.cache_to_disk(k, v, item)

        self._cache_items[item] = True
        return arrays

    def cache_to_disk(self, key, value, item):
        name = self.d.filename(item)
        filepath: Path = self.cache_folders[key].cache_folder / str(name)
        if self.cache_folders[key].is_image:
            save_image(value, str(filepath.with_suffix(".jpg")))
        else:
            np.save(filepath.with_suffix(".npy"), value)

    def get_cache_folder(self) -> Path:
        root_img = Path(self.d.img_filepath["image"][0]).parent
        folder_name = root_img.name
        cache_folder = root_img.parent / self.cache_dir / f".{folder_name}_cache" / self.d.id
        return cache_folder

    def remap(self, old_key: str, new_key: str):
        for k, v in self.cache_folders.items():
            if k == old_key:
                self.cache_folders[new_key] = v
                del self.cache_folders[k]
        if old_key in self.in_memory_items:
            self.in_memory_items.remove(old_key)

        self.in_memory_items.append(new_key)