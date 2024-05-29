import json
import pathlib 
class BufferedJsonWriter(object):
    def __init__(self, file_name, buffer_size=50):
        self.file_path = file_name
        self.buffer = []
        self.buffer_size = buffer_size

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.write_buffer()

    def write(self, obj=None):
        if obj is not None:
            self.buffer.append(obj)
        if len(self.buffer)>=self.buffer_size:
            self.write_buffer()

    def write_buffer(self):
        with open(self.file_path, "a") as data_file:
            data_file.write(json.dumps(self.buffer))
            data_file.write("\n")
            self.buffer = []

class BufferedJsonReader(object):
    def __init__(self, file_name):
        self.file_path = file_name

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __itr__(self):
        with open(self.file_path, "r") as data_file:
            for line in data_file:
                yield from json.loads(line)
                
    def read(self):
        return list(self.__itr__())



def get_cache_path(dataset):
    cache_files = dataset.cache_files
    if isinstance(cache_files,dict):
        cache_files = next(iter(cache_files.values()))
    return pathlib.Path(cache_files[0]['filename']).parent