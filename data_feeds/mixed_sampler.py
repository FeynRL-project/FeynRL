import numpy as np
from torch.utils.data import Sampler, DistributedSampler, ConcatDataset

class MixedDatasetSampler(Sampler):
    def __init__(self, seed: int,
                       dnames: list[str], # ['d1', 'd2', 'd3']
                       ratios: dict[str, float], # {'d1':0.3, 'd2':0.1, 'd3':0.6}
                       local_batch_size: int, # local_bsz = global_bsz // world_size, this is done outside the sampler
                       steps_per_epoch: int, # Steps per rank not global
                       len_datasets: dict[str, int], # {'d1':256, 'd2':4545, 'd3':1545}
                       shuffle_within_batch: bool,
                       dynamic_ratio_every_step: bool=True,
                       world_size :int=1,
                       rank: int=0):
        assert len(len_datasets) == len(ratios) == len(dnames), "len_datasets, ratios, and dnames must have the same length"
        if set(dnames) != set(ratios.keys()):
            raise ValueError(f"Dataset/ratio key mismatch: missing={set(dnames)-set(ratios.keys())}, "
                             f"extra={set(ratios.keys())-set(dnames)}")
        assert world_size >= 1 and rank >= 0, "world_size must be >= 1 and rank must be >= 0"
        assert rank < world_size, "rank must be less than world_size"

        # Check against empty datasets
        for name, length in len_datasets.items():
            if length == 0:
                raise ValueError(f"Dataset '{name}' is empty. Cannot sample from it.")

        ########
        # 1. Generic setup (e.g., random seed, device, world size, etc.)
        ########
        self.seed = seed
        self.dnames = dnames
        self.len_datasets = len_datasets
        self.num_datasets = len(dnames)

        self.shuffle_within_batch = shuffle_within_batch
        self.local_batch_size = local_batch_size
        self.steps_per_epoch = steps_per_epoch

        self.world_size = world_size
        self.rank = rank

        ########
        # 2. sampler related configs
        ########
        self.epoch = 0 # we need this to reshuffle the datasets in each epoch
        self.rng = np.random.default_rng(self.seed + self.rank)

        # It is important and critical to fix the order of datasets as we use ConcatDataset to mix datasets
        # ratio is : {'d1':0.3, 'd2':0.1, 'd3':0.6}
        self.ratios_list = np.array([ratios[name] for name in self.dnames], dtype=np.float32)
        self.probs = self.ratios_list / self.ratios_list.sum()
        assert  self.ratios_list.sum() > 0, "sum of ratios must be positive"
        assert  self.ratios_list.min() > 0, "all ratios must be positive"

        ########
        # 3. Data Concat related configs
        ########

        # offsets are the starting index of each dataset in the concatenated dataset.
        # for example, index from dataset i will be i + offsets[i]
        self.offsets = {}
        curr_offset = 0
        for name in self.dnames:
            self.offsets[name] = curr_offset
            curr_offset += self.len_datasets[name]

        ########
        # 4. Dynamic/static ratio related setup
        ########
        self.redo_ratio_every_step = (self.num_datasets > self.local_batch_size) or (dynamic_ratio_every_step == True)
        if self.redo_ratio_every_step == False:
            self.sample_per_dataset = self._fixed_sample_count()

    def _fixed_sample_count(self):
        '''
            Stable rounding to make sure the sum of the ratios is equal to the batch size. This 
            method also returns same ratio hence it is fixed ratio and needs to be called just once. 
            This is largest remainder method appraoch.
        '''
        raw_samples = self.probs * self.local_batch_size
        base_samples = np.floor(raw_samples).astype(int)

        # add the remaining samples to the dataset with the highest ratio
        remaining_samples = self.local_batch_size - np.sum(base_samples)
        if remaining_samples > 0:
            frac = raw_samples - base_samples
            # largest fractional parts first
            # we want largest fractional parts so we look at the end of the sorted array
            order = np.argsort(frac)[::-1] # descending order
            base_samples[order[:remaining_samples]] += 1
        out_counts = {name: int(count) for name, count in zip(self.dnames, base_samples)}

        return out_counts

    def _probabilistic_sample_counts(self):
        '''
            calculate the sampling counts for each dataset
            this is done based on the ratio of each dataset and samples in each dataset
        '''
        # self.probs is an ordered list per dataset as it was fixed in the init
        # if data set is as ['d1', 'd2', 'd3'], then probs is [probs(d1), probs(d2), probs(d3)]
        selected_datasets_ratio = self.rng.multinomial(self.local_batch_size, self.probs).tolist()
        out_counts = {name: int(count) for name, count in zip(self.dnames, selected_datasets_ratio)}
        return out_counts

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            # each rank just samples independently with a rank-seeded rng
            # this is important for distributed training
            batch = []
            # recalculate sample count if need
            sample_per_dataset = self._probabilistic_sample_counts() if self.redo_ratio_every_step else self.sample_per_dataset
            
            for current_dataset_name, count in sample_per_dataset.items():
                if count == 0:
                    continue

                # sample from the dataset
                # self.rng.integers implies sampling with replacement.
                local_indices = self.rng.integers(low=0, high=self.len_datasets[current_dataset_name], size=count)
                global_indices = [self.offsets[current_dataset_name] + i  for i in local_indices]
                batch.extend(global_indices) 

            if self.shuffle_within_batch:
                self.rng.shuffle(batch)
            yield batch    

    def __len__(self):
        '''
            return the number of steps per epoch per rank
        '''
        return self.steps_per_epoch

    def set_epoch(self, epoch: int):
        '''
            set the epoch number at the start of each epoch to reshuffle the datasets
        '''
        self.epoch = epoch
        # Use SeedSequence for robust, independent streams across ranks and epochs
        self.rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, self.rank)))

def create_dataset_and_sampler(data_paths,
                               prompt_key,
                               answer_key,
                               max_seq_len,
                               tokenizer,
                               train_ratios,
                               split,
                               rank,
                               world_size,
                               seed,
                               local_batch_size,
                               dataset_cls,
                               dynamic_ratio_every_step,
                               steps_per_epoch=None,
                               shuffle_within_batch=True,
                               ):
    '''
        This function inits data loader per dataset and returns a concat dataset.
        For validation, train_ratios is ignored and we use DistributedSampler.
    '''
    all_datasets = []
    dname_list = []
    len_datasets = {}
    for d_path in data_paths:
        if rank == 0:
            print(f"Loading dataset from {d_path}")
        # extract the dataset name from the file path
        dname = d_path.split("/")[-1].split(".")[0]
        dname_list.append(dname)
        # load each dataset
        dataset = dataset_cls(prompt_key=prompt_key,
                             answer_key=answer_key,
                             max_seq_len=max_seq_len,
                             tokenizer=tokenizer,
                             data_path=d_path)
        all_datasets.append(dataset)
        len_datasets[dname] = len(dataset)

    # now concat all datasets
    concat_ds = ConcatDataset(all_datasets)

    if split == 'train':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for training split")
        sampler = MixedDatasetSampler(seed=seed,
                                      dnames=dname_list,
                                      ratios=train_ratios,
                                      len_datasets=len_datasets,
                                      local_batch_size=local_batch_size,
                                      steps_per_epoch=steps_per_epoch,
                                      rank=rank,
                                      world_size=world_size,
                                      shuffle_within_batch=shuffle_within_batch,
                                      dynamic_ratio_every_step=dynamic_ratio_every_step)
    else:
        # For validation, DistributedSampler is used to ensure every sample is seen exactly once.
        sampler = DistributedSampler(concat_ds,
                                     shuffle=False,
                                     num_replicas=world_size,
                                     rank=rank,
                                     drop_last=False)

    return concat_ds, sampler

def create_prompt_dataset_and_sampler(data_paths,
                                      prompt_key,
                                      solution_key,
                                      max_seq_len,
                                      tokenizer,
                                      train_ratios,
                                      seed,
                                      local_batch_size,
                                      dataset_cls,
                                      dynamic_ratio_every_step,
                                      steps_per_epoch,
                                      shuffle_within_batch=True,
                                      ):
    '''
        Creates a concat dataset and MixedDatasetSampler for rollout generation.
    '''
    all_datasets = []
    dname_list = []
    len_datasets = {}
    for d_path in data_paths:
        print(f"Loading dataset from {d_path}")
        # extract the dataset name from the file path
        dname = d_path.split("/")[-1].split(".")[0]
        dname_list.append(dname)
        # load each dataset
        dataset = dataset_cls(prompt_key=prompt_key,
                              solution_key=solution_key,
                              max_seq_len=max_seq_len,
                              tokenizer=tokenizer,
                              data_path=d_path)

        all_datasets.append(dataset)
        len_datasets[dname] = len(dataset)

    # concat all datasets
    concat_ds = ConcatDataset(all_datasets)
    # since rollout dataloader runs on the orchestrator,
    # we set rank=0 and world_size=1
    sampler = MixedDatasetSampler(seed=seed,
                                  dnames=dname_list,
                                  ratios=train_ratios,
                                  len_datasets=len_datasets,
                                  local_batch_size=local_batch_size,
                                  steps_per_epoch=steps_per_epoch,
                                  rank=0,
                                  world_size=1,
                                  shuffle_within_batch=shuffle_within_batch,
                                  dynamic_ratio_every_step=dynamic_ratio_every_step)

    # all datasets have the same collate_fn since they are same type
    collate_fn = all_datasets[0].collate_fn

    return concat_ds, sampler, collate_fn