import torch.utils.data as data
import torch
import os
import logging

from PIL import Image
from timm.data import IterableImageDataset, ImageDataset

from timm.data.parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class StoredImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0
        self.storage = {}

    def __getitem__(self, index):
        if index not in self.storage:
            img, target = self.parser[index]
            try:
                img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
            except Exception as e:
                _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
                self._consecutive_errors += 1
                if self._consecutive_errors < _ERROR_RETRY:
                    return self.__getitem__((index + 1) % len(self.parser))
                else:
                    raise e
            self._consecutive_errors = 0
            if self.transform is not None:
                img = self.transform(img)
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
        else:
            img, target = self.storage[index]
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    name = name.lower()
    if name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        if kwargs.pop('fixed_aug', False):
            _logger.info('creating fixed aug dataset')
            ds = StoredImageDataset(root, parser=name, **kwargs)
        else:
            ds = ImageDataset(root, parser=name, **kwargs)
    return ds
