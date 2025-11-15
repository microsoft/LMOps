import sys, os
import json


def run_pulldata():
    print("Running pull data")
    # sys.exit(0)
    input_file = sys.argv[1]
    results = {}
    eval_task = ['aime24', 'math500', 'amc23', 'minerva_math', 'olympiadbench']
    subfolders = [f.path for f in os.scandir(input_file) if f.is_dir()]
    for folder in subfolders:
        if 'global_step' not in folder:
            continue

        step = folder.split('/')[-1].replace('global_step_', '')
        results[step] = {}
        for task in eval_task:
            task_folder = os.path.join(folder, task)
            if not os.path.exists(task_folder):
                continue
            for file in os.listdir(task_folder):
                if 'metric' in file:
                    # load json file
                    _obj = json.load(open(os.path.join(task_folder, file)))
                    results[step][task] = _obj['acc']
                else:

                    # output results, count the response length
                    if 'length' not in results[step]:
                        results[step]['length'] = []
                    with open(os.path.join(task_folder, file), 'r') as f:
                        for line in f:
                            _obj = json.loads(line)
                            if 'code' in _obj:
                                results[step]['length'].append(len(_obj['code'][0].split(' ')))
        if len([results[step][task] for task in eval_task if task in results[step]]) != 0:
            results[step]['avg'] = round(sum([results[step][task] for task in eval_task if task in results[step]]) / len([results[step][task] for task in eval_task if task in results[step]]), 2)
            results[step]['length'] = round(sum(results[step]['length']) / len(results[step]['length']), 2)

    # print results as table
    # sort results by step
    gamma = ';'
    results = dict(sorted(results.items(), key=lambda x: int(x[0])))
    print('step', end=gamma)
    for task in eval_task + ['avg', 'length']:
        print(task, end=gamma)
    print()
    for step in results:
        if len(results[step]) == 0:
            continue
        print(step, end=gamma)
        for task in eval_task + ['avg', 'length']:
            if task in results[step]:
                print(results[step][task], end=gamma)
            else:
                print('N/A', end=gamma)
        print()
    return results

if __name__ == '__main__':
    run_pulldata()
# python pull_results.py /mnt/msranlp/shaohanh/exp/simple-rl/checkpoints/Qwen2.5-Math-7B_ppo_from_base_math_lv35_2node/_actor/