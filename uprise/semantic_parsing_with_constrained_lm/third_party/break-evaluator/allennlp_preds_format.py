import json
import re

def main():
    # todo: @@SEP@@ to ; , @@#@@ to #
    predictions_file = "old_data_dev_low_level_preds.json"
    traget_file= predictions_file.replace('.json', '.csv')
    with open(predictions_file, "r") as fd:
        preds = [json.loads(line) for line in fd.readlines()]
    preds = [re.sub(r'@@(\d+)@@', '#\g<1>', re.sub('@@SEP@@',';', ' '.join(p['predicted_tokens'][0]))) for p in preds]
    preds.insert(0,'prediction')
    preds = [f'"{p}"\n' for p in preds]
    with open(traget_file, "wt") as fd:
        fd.writelines(preds)


if __name__ == '__main__':
    main()