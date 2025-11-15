import os
import json
import glob
import argparse
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import matplotlib.pyplot as plt
import re

# Create a thread-local storage for tokenizer
thread_local = threading.local()

def get_tokenizer(model_name):
    """Get or create thread-local tokenizer"""
    if not hasattr(thread_local, 'tokenizer'):
        thread_local.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return thread_local.tokenizer

def normalize_model_name(path):
    """Extract and normalize model name from path"""
    parts = path.split('/')
    # First check for checkpoint pattern
    for part in parts[::-1]:
        if 'checkpoint' in part:
            idx = parts.index(part)
            model_name = parts[idx-1]
            checkpoint = part
            return f"{model_name}-{checkpoint}"
        # Add check for global_step pattern
        if 'global_step' in part:
            idx = parts.index(part)
            model_name = parts[idx-1]
            return f"{model_name}-{part}"
    
    # If no checkpoint or global_step found, use the last meaningful part and add checkpoint-final
    for part in reversed(parts):
        if any(x in part.lower() for x in ['llama', 'qwen', 'gpt', 'mistral']):
            return f"{part}-checkpoint-final"
    
    return "unknown_model"

def get_benchmark_name(path):
    """Extract benchmark name from path"""
    parts = path.split('/')
    # Look for common benchmark names in the path
    # for part in parts:
    #     if part.lower() in ['aime24', 'gsm8k', 'math500']:
    #         return part.lower()
    #TODO: potential bug for diff path
    return parts[-2]
    # return "unknown_benchmark"

def get_jsonl_path(metrics_file):
    """Get corresponding jsonl file path"""
    # Get the directory containing the metrics file
    metric_folder = os.path.dirname(metrics_file)
    
    # The JSONL file should be in the same directory with a .jsonl extension
    # and without the '_metrics' suffix
    base_name = os.path.basename(metrics_file).replace('_metrics.json', '')
    jsonl_file = os.path.join(metric_folder, f"{base_name}.jsonl")
    
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
    
    return jsonl_file

def calculate_avg_tokens(jsonl_path, tokenizer):
    """Calculate average tokens in the first code element"""
    if not os.path.exists(jsonl_path):
        print(f"Warning: JSONL file not found: {jsonl_path}")
        return 0
        
    total_tokens = 0
    count = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'code' in data and isinstance(data['code'], list) and len(data['code']) > 0:
                    tokens = len(tokenizer.encode(data['code'][0]))
                    total_tokens += tokens
                    count += 1
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")
        return 0
        
    return total_tokens / count if count > 0 else 0

def process_file(args):
    """Process a single metrics file"""
    metrics_file, model_name = args
    try:
        # Get model and benchmark names
        model_name_norm = normalize_model_name(metrics_file)
        benchmark = get_benchmark_name(metrics_file)
        
        # Read metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            acc = metrics.get('acc', 0)
        
        # Get corresponding jsonl file
        jsonl_file = get_jsonl_path(metrics_file)
        tokenizer = get_tokenizer(model_name)
        avg_tokens = calculate_avg_tokens(jsonl_file, tokenizer)
        
        return model_name_norm, benchmark, {
            'acc': acc,
            'tokens': avg_tokens
        }
        
    except Exception as e:
        print(f"Error processing {metrics_file}: {e}")
        return None

def collect_results(base_dir, model_name, num_threads=8):
    # Initialize results storage
    results = defaultdict(lambda: defaultdict(dict))
    
    # Find all metrics.json files
    metrics_files = glob.glob(f"{base_dir}/**/test_*metrics.json", recursive=True)
    
    # Create arguments for parallel processing
    process_args = [(f, model_name) for f in metrics_files]
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list(tqdm(
            executor.map(process_file, process_args),
            total=len(metrics_files),
            desc="Processing files"
        ))
        
        # Collect results
        for result in futures:
            if result is not None:
                model_name, benchmark, metrics = result
                results[model_name][benchmark] = metrics
    
    return results

def create_summary(results):
    # Convert results to DataFrame
    rows = []
    for model, benchmarks in results.items():
        row = {'model': model}
        total_acc = 0
        total_tokens = 0
        count = 0
        
        for benchmark, metrics in benchmarks.items():
            row[f'{benchmark}_acc'] = metrics['acc']
            row[f'{benchmark}_tokens'] = metrics['tokens']
            total_acc += metrics['acc']
            total_tokens += metrics['tokens']
            count += 1
        
        if count > 0:
            row['avg_acc'] = total_acc / count
            row['avg_tokens'] = total_tokens / count
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort DataFrame by checkpoint/global_step number
    def get_step_number(model_name):
        if 'checkpoint-final' in model_name:
            return float('inf')
        # Check for checkpoint pattern
        checkpoint_match = re.search(r'checkpoint-(\d+)', model_name)
        if checkpoint_match:
            return int(checkpoint_match.group(1))
        # Check for global_step pattern
        global_step_match = re.search(r'global_step(\d+)', model_name)
        if global_step_match:
            return int(global_step_match.group(1))
        return float('inf')
    
    # Sort DataFrame based on step numbers
    df['sort_key'] = df['model'].apply(get_step_number)
    df = df.sort_values('sort_key')
    df = df.drop('sort_key', axis=1)
    
    return df

def sync_to_wandb(args, results, project_name, df, plot_dir, csv_path):
    """Sync results, CSV table and plots to wandb"""
    # Initialize wandb run
    run = wandb.init(
        project=project_name,
        name=args.wandb_run_name,
        reinit=True
    )
    
    # Log the CSV table as a wandb Table
    table = wandb.Table(dataframe=df)
    wandb.log({"results_table": table})
    
    # Also save the CSV file as an artifact
    artifact = wandb.Artifact('evaluation_results', type='dataset')
    artifact.add_file(csv_path)
    run.log_artifact(artifact)
    
    # Log plots
    if os.path.exists(plot_dir):
        for plot_file in os.listdir(plot_dir):
            if plot_file.endswith('_progress.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
    
    run.finish()

def sort_checkpoints(models):
    """Sort checkpoints numerically with final checkpoint at the end"""
    def get_checkpoint_num(model_name):
        if 'checkpoint-final' in model_name:
            return float('inf')
        # Check for checkpoint pattern
        checkpoint_match = re.search(r'checkpoint-(\d+)', model_name)
        if checkpoint_match:
            return int(checkpoint_match.group(1))
        # Check for global_step pattern
        global_step_match = re.search(r'global_step(\d+)', model_name)
        if global_step_match:
            return int(global_step_match.group(1))
        return float('inf')
    
    # Group models by base name (everything before checkpoint- or global_step)
    model_groups = defaultdict(list)
    for model in models:
        # Split on either checkpoint- or global_step
        base_name = re.split(r'(?:checkpoint-|global_step)', model)[0].rstrip('-')
        model_groups[base_name].append(model)
    
    # Sort each group's checkpoints
    sorted_models = []
    for base_name, checkpoints in model_groups.items():
        sorted_checkpoints = sorted(checkpoints, key=get_checkpoint_num)
        sorted_models.extend(sorted_checkpoints)
    
    return sorted_models

def plot_training_progress(results, output_dir, benchmarks=None):
    """Plot training progress for each model series"""
    # Get all unique benchmarks
    all_benchmarks = set()
    for model_metrics in results.values():
        all_benchmarks.update(model_metrics.keys())
    all_benchmarks = sorted(list(all_benchmarks))
    
    # Filter benchmarks if specified
    if benchmarks:
        all_benchmarks = [b for b in all_benchmarks if b in benchmarks]
    
    # Group models by base name
    model_groups = defaultdict(list)
    for model in results.keys():
        # Use the same splitting logic as in sort_checkpoints
        base_name = re.split(r'(?:checkpoint-|global_step)', model)[0].rstrip('-')
        model_groups[base_name].append(model)
    
    # Create plots for each model group
    for base_name, models in model_groups.items():
        if len(models) <= 1:
            continue
            
        # Sort checkpoints
        models = sort_checkpoints(models)
        
        # Extract checkpoint numbers for x-axis
        checkpoints = []
        for model in models:
            if 'checkpoint-final' in model:
                checkpoints.append('final')
            else:
                # Check for checkpoint pattern
                checkpoint_match = re.search(r'checkpoint-(\d+)', model)
                if checkpoint_match:
                    checkpoints.append(checkpoint_match.group(1))
                    continue
                # Check for global_step pattern
                global_step_match = re.search(r'global_step(\d+)', model)
                if global_step_match:
                    checkpoints.append(f'step{global_step_match.group(1)}')
                else:
                    checkpoints.append('unknown')
        
        # Create figure with subplots
        n_benchmarks = len(all_benchmarks) + 1  # +1 for average
        n_cols = 3
        n_rows = (n_benchmarks + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle(f'Training Progress - {base_name}')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot average metrics first
        avg_acc = []
        avg_tok = []
        for model in models:
            metrics = results[model]
            acc_values = [metrics[b]['acc'] for b in all_benchmarks if b in metrics]
            tok_values = [metrics[b]['tokens'] for b in all_benchmarks if b in metrics]
            avg_acc.append(sum(acc_values) / len(acc_values) if acc_values else 0)
            avg_tok.append(sum(tok_values) / len(tok_values) if tok_values else 0)
        
        # Create twin axis for the first plot
        ax_tok = axes[0].twinx()
        
        # Plot accuracy and tokens (in the average plot)
        line_acc = axes[0].plot(range(len(checkpoints)), avg_acc, marker='o', 
                              color='#1f77b4', label='Accuracy')  # 浅蓝色
        line_tok = ax_tok.plot(range(len(checkpoints)), avg_tok, marker='s', 
                             color='#ff7f0e', label='Tokens')  # 橙色
        
        # Set labels and title
        axes[0].set_title('Average Metrics')
        axes[0].set_xlabel('Checkpoint')
        axes[0].set_ylabel('Accuracy (%)', color='#1f77b4')  # 浅蓝色
        ax_tok.set_ylabel('Tokens', color='#ff7f0e')  # 橙色
        
        # Set ticks
        axes[0].set_xticks(range(len(checkpoints)))
        axes[0].set_xticklabels(checkpoints, rotation=45)
        
        # Add grid
        axes[0].grid(True, alpha=0.3)
        
        # Add value annotations with matching colors
        for i, (v_acc, v_tok) in enumerate(zip(avg_acc, avg_tok)):
            axes[0].annotate(f'{v_acc:.1f}', 
                           (i, v_acc), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           color='#1f77b4',  # 浅蓝色
                           fontsize=8)  # 减小字体大小
            ax_tok.annotate(f'{v_tok:.1f}', 
                          (i, v_tok), 
                          textcoords="offset points", 
                          xytext=(0,-15), 
                          ha='center',
                          color='#ff7f0e',  # 橙色
                          fontsize=8)  # 减小字体大小
        
        # Add legend
        lines = line_acc + line_tok
        labels = [l.get_label() for l in lines]
        axes[0].legend(lines, labels, loc='upper left')
        
        # Plot individual benchmarks
        for i, benchmark in enumerate(all_benchmarks, start=1):
            # Collect metrics
            acc_values = []
            tok_values = []
            for model in models:
                metrics = results[model]
                acc_values.append(metrics.get(benchmark, {}).get('acc', 0))
                tok_values.append(metrics.get(benchmark, {}).get('tokens', 0))
            
            # Create twin axis
            ax_tok = axes[i].twinx()
            
            # Plot accuracy and tokens (in the individual plot)
            line_acc = axes[i].plot(range(len(checkpoints)), acc_values, marker='o', 
                                  color='#1f77b4', label='Accuracy')  # 浅蓝色
            line_tok = ax_tok.plot(range(len(checkpoints)), tok_values, marker='s', 
                                 color='#ff7f0e', label='Tokens')  # 橙色
            
            # Set labels with matching colors
            axes[i].set_title(f'{benchmark}')
            axes[i].set_xlabel('Checkpoint')
            axes[i].set_ylabel('Accuracy (%)', color='#1f77b4')  # 浅蓝色
            ax_tok.set_ylabel('Tokens', color='#ff7f0e')  # 橙色
            
            # Set ticks
            axes[i].set_xticks(range(len(checkpoints)))
            axes[i].set_xticklabels(checkpoints, rotation=45)
            
            # Add grid
            axes[i].grid(True, alpha=0.3)
            
            # Add value annotations with matching colors
            for j, (v_acc, v_tok) in enumerate(zip(acc_values, tok_values)):
                axes[i].annotate(f'{v_acc:.1f}', 
                               (j, v_acc), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center',
                               color='#1f77b4',  # 浅蓝色
                               fontsize=8)  # 减小字体大小
                ax_tok.annotate(f'{v_tok:.1f}', 
                              (j, v_tok), 
                              textcoords="offset points", 
                              xytext=(0,-15), 
                              ha='center',
                              color='#ff7f0e',  # 橙色
                              fontsize=8)  # 减小字体大小
            
            # Add legend
            lines = line_acc + line_tok
            labels = [l.get_label() for l in lines]
            axes[i].legend(lines, labels, loc='upper left')
        
        # Remove empty subplots
        for i in range(len(all_benchmarks) + 1, len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout and save
        fig.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f'{base_name}_progress.png')
        
        # Delete existing file if it exists
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
            except Exception as e:
                print(f"Warning: Could not remove existing file {output_filename}: {e}")
            
        try:
            fig.savefig(output_filename)
            print(f"Saved plot to: {output_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close(fig)

def main(args):
    base_dir = args.base_dir
    model_name = args.model_name
    
    # Parse benchmarks if specified
    benchmarks = None
    if args.benchmarks:
        benchmarks = set(args.benchmarks.split(','))
    
    # Collect results
    print("Collecting results...")
    results = collect_results(base_dir, model_name, args.num_threads)
    
    # Filter results if benchmarks specified
    if benchmarks:
        filtered_results = defaultdict(lambda: defaultdict(dict))
        for model, model_results in results.items():
            for benchmark, metrics in model_results.items():
                if benchmark in benchmarks:
                    filtered_results[model][benchmark] = metrics
        results = filtered_results
    
    # Create summary DataFrame
    print("\nCreating summary...")
    df = create_summary(results)
    print("\nResults summary:")
    print(df)
    
    # Plot training progress
    print("\nCreating training progress plots...")
    plot_training_progress(results, args.plot_dir, benchmarks)
    
    # Save to CSV
    output_file = args.output_path
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Sync to wandb if enabled
    if args.use_wandb:
        print("\nSyncing to wandb...")
        if args.wandb_api_key:
            wandb.login(key=args.wandb_api_key)
        sync_to_wandb(args, results, args.wandb_project, df, args.plot_dir, args.output_path)
        print("Wandb sync completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="outputs/project/reasonshort/weiliu/cot/output")
    parser.add_argument("--model_name", type=str, default="Qwen-math-7B-S100-qwq-fs-7k8-8192len-5e-6-rope10-bsz64")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="math-eval-results")
    parser.add_argument("--wandb_api_key", type=str, default="1635b1d5d43c5ca1cb6f0b22aa8b0960e0491c52")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--benchmarks", type=str, 
                       default="gsm8k,math,minerva_math,olympiadbench,college_math,aime24,amc23",
                       help="Comma-separated list of benchmarks to include")
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = os.path.join(args.base_dir, "eval_results.csv")
    
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.base_dir, "plots")
        
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir, exist_ok=True)
        
    main(args)
