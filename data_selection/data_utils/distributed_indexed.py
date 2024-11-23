# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import struct

import numpy as np
import torch
import torch.distributed as dist


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


class DistributedMMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'
        def __init__(self, path):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, name, rank_number=0, rank_total=1, do_probe=True, 
                 min_state=0, max_state=None, min_offset=0, max_offset=None, min_ratio=None, max_ratio=None,
                 cache = None, load_to_ram=False):
        
        super().__init__()

        self._path = path
        self._name = name
        self._do_probe = do_probe
        self._load_to_ram = load_to_ram
        self._state = min_state
        self.min_state = min_state
        self.min_offset = min_offset
        self.max_offset = max_offset
        if cache is not None:
            self._cache = cache
            os.makedirs(self._cache, exist_ok=True)
        else:
            self._cache = None
        self._rank_total = rank_total
        self._rank_number = rank_number
        self._index = None
        self._bin_buffer = None
        self._bin_buffer_mmap = None
        self.max_state, self.history, self.lens = self._probe_data_path(self._path, self._name, self._rank_total, do_probe=do_probe, min_state=min_state, max_state=max_state)
        self.total_length = int(self.history[self.max_state-1][1])

        if min_ratio is not None:
            self.min_offset = int(min_ratio * self.total_length)
        
        if max_ratio is not None:
            self.max_offset = int(max_ratio * self.total_length)

        self.valid_length = min((self.max_offset if self.max_offset is not None else self.total_length), self.total_length) - self.min_offset

        if not dist.is_initialized() or dist.get_rank() == 0:  
            print(f"Probing end. Max data state {self.max_state}, total length {self.history[self.max_state-1][1]}, valid_length {self.valid_length}, min_offset {self.min_offset}")

        self._do_init(self._path, self._name, self._cache, self._state, do_probe, load_to_ram)

    def _probe_data_path(self, path, name, rank_total, do_probe, min_state, max_state):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Probing Dataset")
        history = {min_state-1:(0, 0)}
        lens = []
        state = min_state
        max_state = np.iinfo(np.int32).max if max_state is None else max_state
        while state < max_state:
            if state % 10 == 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Find data state {state}")
            if do_probe:
                source_file = os.path.join(path, name + f"_{state}")
            else:
                source_file = os.path.join(path, name)

            if self.exists(source_file):
                index = self.Index(index_file_path(source_file))
                history[state] = (history[state-1][1], history[state-1][1] + len(index))
                lens.append(len(index))
            else:
                break

            state += 1
            if not do_probe:
                break
                
        return state, history, lens

    def __getstate__(self):
        return os.path.join(self._path, self._name + "_%d"%(self._state))

    def __setstate__(self, state):
        self._state = state
        self._do_init(self._path, self._name, self._cache, self._state, self._do_probe, self._load_to_ram)

    def _do_init(self, path, name, cache, state, do_probe, load_to_ram):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

        self._state = state

        if do_probe:
            source_file = os.path.join(path, name + f"_{self._state}")
        else:
            source_file = os.path.join(path, name)
        
        assert os.path.exists(data_file_path(source_file)), "Data file not found: {}".format(data_file_path(source_file))
        assert os.path.exists(index_file_path(source_file)), "Index file not found: {}".format(index_file_path(source_file))
        self._index = self.Index(index_file_path(source_file))
        
        if load_to_ram:
            print("Loading from file")
            self._bin_buffer = np.fromfile(data_file_path(source_file), dtype=self._index.dtype)
            print("Loading from file done")    
        else:
            self._bin_buffer_mmap = np.memmap(data_file_path(source_file), mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

    def __len__(self):
        return self.valid_length

    # def _next_file(self):
    #     self._state += 1
    #     if self._state >= self.max_state:
    #         # self._state = 0
    #         raise StopIteration()
    #     # print_rank(f"next_file: {self._state}")
    #     self._do_init(self._path, self._name, self._cache, self._state, self._do_probe, self._load_to_ram)
    
    def __relative_idx(self, idx):
        res = idx - self.history[self._state][0]
        return res

    # def __slice_item(self, start, stop):
    #     ptr = self._index._pointers[self.__relative_idx(start)]
    #     sizes = self._index._sizes[self.__relative_idx(start):self.__relative_idx(stop)]
    #     offsets = list(accumulate(sizes))
    #     np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=sum(sizes), offset=ptr)
    #     return np.split(np_array, offsets[:-1])

    def __getitem__(self, idx):
        idx += self.min_offset

        if isinstance(idx, int):
            if idx >= self.total_length:
                print(f"Distributed index stop interation. Idx: {idx} Total_length: {self.total_length}")
                raise StopIteration
            
            origin_state = self._state
            while idx >= self.history[self._state][1] or idx < self.history[self._state][0]:
                self._state += 1
                if self._state >= self.max_state:
                    self._state = 0
                # print(self._state)
            if self._state != origin_state:
                self._do_init(self._path, self._name, self._cache, self._state, self._do_probe, self._load_to_ram)
            ptr, size = self._index[self.__relative_idx(idx)]
            return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        elif isinstance(idx, slice):
            raise NotImplementedError()
        else:
            raise TypeError("Error type: {}".format(str(type(idx))))

    @property
    def sizes(self):
        return self._index.sizes
        
    def exists(self, path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )
