import os
import shutil
import numpy as np
import argparse
from tqdm import tqdm
from data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder
from arguments import add_data_args, add_runtime_args
from numerize.numerize import numerize


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_data_args(add_runtime_args(parser))
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    for max_state in range(1000000):
        if not os.path.exists(os.path.join(args.data_dir, f"data_{max_state}.idx")):
            break

    num = args.dev_num + args.test_num
    
    state = max_state - 1

    print("Max state:", state)
    
    tot_num = 0
    while state >= 0 and tot_num < num:
        data = DistributedMMapIndexedDataset(args.data_dir, f"data_{state}", do_probe=False)
        tot_num += len(data)
        state -= 1
    min_state = state + 1
    print("Min state:", min_state)
    
    all_data = DistributedMMapIndexedDataset(args.data_dir, f"data", min_state=min_state, do_probe=True)

    dtype = all_data[0].dtype.type

    output_path_dev = os.path.join(args.save, args.data_name, f"dev-{numerize(args.dev_num)}")
    output_path_test = os.path.join(args.save, args.data_name, f"test-{numerize(args.test_num)}")
    output_path_trn_left = args.data_dir
    os.makedirs(output_path_dev, exist_ok=True)
    os.makedirs(output_path_test, exist_ok=True)

    chunked_builder_dev = ChunkedDatasetBuilder(args.base_path, output_path_dev, dtype)
    chunked_builder_test = ChunkedDatasetBuilder(args.base_path, output_path_test, dtype)
    chunked_builder_trn_left = ChunkedDatasetBuilder(args.base_path, output_path_trn_left, dtype, split="train_left")

    num_trn_left = len(all_data) - args.dev_num - args.test_num

    print("Num train data left:", num_trn_left)

    offset = 0
    while offset < num_trn_left:
        chunked_builder_trn_left.add_np_item(all_data[offset])
        offset += 1
    chunked_builder_trn_left.finalize()
    
    while offset < num_trn_left + args.dev_num:
        chunked_builder_dev.add_np_item(all_data[offset])
        offset += 1
    chunked_builder_dev.finalize()
    
    while offset < num_trn_left + args.dev_num + args.test_num:
        chunked_builder_test.add_np_item(all_data[offset])
        offset += 1
    chunked_builder_test.finalize()
    
    # move old data to backup
    for s in range(min_state, max_state):
        shutil.move(os.path.join(args.data_dir, f"data_{s}.bin"), os.path.join(args.data_dir, f"data_{s}.bin.bak"))
        shutil.move(os.path.join(args.data_dir, f"data_{s}.idx"), os.path.join(args.data_dir, f"data_{s}.idx.bak"))

    print("state:", min_state)
    shutil.copy(os.path.join(args.data_dir, f"train_left_0.bin"), os.path.join(args.data_dir, f"data_{min_state}.bin"))
    shutil.copy(os.path.join(args.data_dir, f"train_left_0.idx"), os.path.join(args.data_dir, f"data_{min_state}.idx"))
    

if __name__ == "__main__":
    main()