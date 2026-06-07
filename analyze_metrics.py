import pandas as pd
import numpy as np
from itertools import combinations
import os
import glob

def load_and_compute_metrics(csv_path):
    """Load CSV file and compute mean SMI and TDI values"""
    df = pd.read_csv(csv_path)
    
    # Drop rows where both smi and tdi are NaN
    df_metrics = df.dropna(subset=['smi', 'tdi'], how='all')
    
    # Compute mean values
    mean_smi = df_metrics['smi'].mean()
    mean_tdi = df_metrics['tdi'].mean()
    
    return mean_smi, mean_tdi

def discover_algorithms(datasets):
    """Automatically discover all available algorithms from distribution directory"""
    algorithms = set()
    for dataset in datasets:
        pattern = f'distribution/*_{dataset}.csv'
        files = glob.glob(pattern)
        for file in files:
            # Extract algorithm name from filename like 'fedavg_cifar10.csv'
            basename = os.path.basename(file)
            algo_name = basename.replace(f'_{dataset}.csv', '')
            algorithms.add(algo_name)
    return sorted(list(algorithms))

def main():
    # Define datasets
    datasets = ['cifar10', 'cifar100']
    
    # Automatically discover all available algorithms
    algorithms = discover_algorithms(datasets)
    print(f"Discovered {len(algorithms)} algorithms: {', '.join(algorithms)}\n")
    
    # Store results
    results = {}
    
    # Load and compute metrics for each algorithm-dataset combination
    for dataset in datasets:
        results[dataset] = {}
        for algo in algorithms:
            csv_path = f'distribution/{algo}_{dataset}.csv'
            try:
                mean_smi, mean_tdi = load_and_compute_metrics(csv_path)
                results[dataset][algo] = {
                    'smi': mean_smi,
                    'tdi': mean_tdi
                }
                print(f"Loaded {csv_path}: SMI={mean_smi:.4f}, TDI={mean_tdi:.4f}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                results[dataset][algo] = None
    
    # Generate Markdown tables
    print("\n" + "="*80)
    print("SMI (Stability Metric Index) - Mean Values")
    print("="*80)
    
    for dataset in datasets:
        print(f"\n### {dataset.upper()} - SMI Mean Values\n")
        print("| Algorithm | Mean SMI |")
        print("|-----------|----------|")
        for algo in algorithms:
            if results[dataset][algo] is not None:
                print(f"| {algo.upper()} | {results[dataset][algo]['smi']:.4f} |")
    
    print("\n" + "="*80)
    print("TDI (Task Drift Index) - Mean Values")
    print("="*80)
    
    for dataset in datasets:
        print(f"\n### {dataset.upper()} - TDI Mean Values\n")
        print("| Algorithm | Mean TDI |")
        print("|-----------|----------|")
        for algo in algorithms:
            if results[dataset][algo] is not None:
                print(f"| {algo.upper()} | {results[dataset][algo]['tdi']:.4f} |")
    
    print("\n" + "="*80)
    print("Pairwise Mean Differences - SMI")
    print("="*80)
    
    for dataset in datasets:
        print(f"\n### {dataset.upper()} - SMI Pairwise Differences\n")
        print("| Algorithm Pair | Mean SMI Difference |")
        print("|----------------|---------------------|")
        
        for algo1, algo2 in combinations(algorithms, 2):
            if results[dataset][algo1] and results[dataset][algo2]:
                diff = abs(results[dataset][algo1]['smi'] - results[dataset][algo2]['smi'])
                print(f"| {algo1.upper()} vs {algo2.upper()} | {diff:.4f} |")
    
    print("\n" + "="*80)
    print("Pairwise Mean Differences - TDI")
    print("="*80)
    
    for dataset in datasets:
        print(f"\n### {dataset.upper()} - TDI Pairwise Differences\n")
        print("| Algorithm Pair | Mean TDI Difference |")
        print("|----------------|---------------------|")
        
        for algo1, algo2 in combinations(algorithms, 2):
            if results[dataset][algo1] and results[dataset][algo2]:
                diff = abs(results[dataset][algo1]['tdi'] - results[dataset][algo2]['tdi'])
                print(f"| {algo1.upper()} vs {algo2.upper()} | {diff:.4f} |")
    
    # Generate comprehensive comparison table
    print("\n" + "="*80)
    print("Comprehensive Comparison Table")
    print("="*80)
    
    print("\n### CIFAR-10 - Complete Metrics Comparison\n")
    print("| Algorithm | Mean SMI | Mean TDI |")
    print("|-----------|----------|----------|")
    for algo in algorithms:
        if results['cifar10'][algo] is not None:
            print(f"| {algo.upper()} | {results['cifar10'][algo]['smi']:.4f} | {results['cifar10'][algo]['tdi']:.4f} |")
    
    print("\n### CIFAR-100 - Complete Metrics Comparison\n")
    print("| Algorithm | Mean SMI | Mean TDI |")
    print("|-----------|----------|----------|")
    for algo in algorithms:
        if results['cifar100'][algo] is not None:
            print(f"| {algo.upper()} | {results['cifar100'][algo]['smi']:.4f} | {results['cifar100'][algo]['tdi']:.4f} |")
    
    # Save results to markdown file
    with open('distribution/metrics_comparison.md', 'w') as f:
        f.write("# SMI and TDI Metrics Comparison\n\n")
        f.write("## Overview\n\n")
        f.write("This report compares the mean SMI (Stability Metric Index) and TDI (Task Drift Index) \n")
        f.write("values across three federated learning algorithms on CIFAR-10 and CIFAR-100 datasets.\n\n")
        
        f.write("## Mean Values\n\n")
        
        for dataset in datasets:
            f.write(f"### {dataset.upper()} - SMI Mean Values\n\n")
            f.write("| Algorithm | Mean SMI |\n")
            f.write("|-----------|----------|\n")
            for algo in algorithms:
                if results[dataset][algo] is not None:
                    f.write(f"| {algo.upper()} | {results[dataset][algo]['smi']:.4f} |\n")
            f.write("\n")
            
            f.write(f"### {dataset.upper()} - TDI Mean Values\n\n")
            f.write("| Algorithm | Mean TDI |\n")
            f.write("|-----------|----------|\n")
            for algo in algorithms:
                if results[dataset][algo] is not None:
                    f.write(f"| {algo.upper()} | {results[dataset][algo]['tdi']:.4f} |\n")
            f.write("\n")
        
        f.write("## Pairwise Mean Differences\n\n")
        
        for dataset in datasets:
            f.write(f"### {dataset.upper()} - SMI Pairwise Differences\n\n")
            f.write("| Algorithm Pair | Mean SMI Difference |\n")
            f.write("|----------------|---------------------|\n")
            
            for algo1, algo2 in combinations(algorithms, 2):
                if results[dataset][algo1] and results[dataset][algo2]:
                    diff = abs(results[dataset][algo1]['smi'] - results[dataset][algo2]['smi'])
                    f.write(f"| {algo1.upper()} vs {algo2.upper()} | {diff:.4f} |\n")
            f.write("\n")
            
            f.write(f"### {dataset.upper()} - TDI Pairwise Differences\n\n")
            f.write("| Algorithm Pair | Mean TDI Difference |\n")
            f.write("|----------------|---------------------|\n")
            
            for algo1, algo2 in combinations(algorithms, 2):
                if results[dataset][algo1] and results[dataset][algo2]:
                    diff = abs(results[dataset][algo1]['tdi'] - results[dataset][algo2]['tdi'])
                    f.write(f"| {algo1.upper()} vs {algo2.upper()} | {diff:.4f} |\n")
            f.write("\n")
        
        f.write("## Comprehensive Comparison\n\n")
        
        for dataset in datasets:
            f.write(f"### {dataset.upper()} - Complete Metrics Comparison\n\n")
            f.write("| Algorithm | Mean SMI | Mean TDI |\n")
            f.write("|-----------|----------|----------|\n")
            for algo in algorithms:
                if results[dataset][algo] is not None:
                    f.write(f"| {algo.upper()} | {results[dataset][algo]['smi']:.4f} | {results[dataset][algo]['tdi']:.4f} |\n")
            f.write("\n")
    
    print(f"\nResults saved to distribution/metrics_comparison.md")

if __name__ == "__main__":
    main()
