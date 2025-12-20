import random
import torch
from torch.utils.data.dataloader import DataLoader, default_collate


def _ms_collate_fn(batch, scale, dataset):
    """
    Custom collate function that handles multi-scale selection for training.
    """
    # Select random scale for training
    idx_scale = 0
    if len(scale) > 1 and hasattr(dataset, 'train') and dataset.train:
        idx_scale = random.randrange(0, len(scale))
        if hasattr(dataset, 'set_scale'):
            dataset.set_scale(idx_scale)
    
    # Collate the batch
    collated = default_collate(batch)
    
    # Append scale index to the batch
    if isinstance(collated, (list, tuple)):
        collated = list(collated)
        collated.append(idx_scale)
    
    return collated


class MSDataLoader(DataLoader):
    """
    Multi-Scale DataLoader compatible with PyTorch 2.x
    
    This dataloader supports random scale selection during training,
    which is useful for multi-scale super-resolution tasks.
    """
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=None, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):

        self.scale = args.scale
        self.dataset = dataset
        
        # Create a wrapper for collate_fn that includes scale handling
        if collate_fn is None:
            collate_fn = default_collate
        
        def wrapped_collate_fn(batch):
            return _ms_collate_fn(batch, self.scale, self.dataset)
        
        super(MSDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=args.n_threads,
            collate_fn=wrapped_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )