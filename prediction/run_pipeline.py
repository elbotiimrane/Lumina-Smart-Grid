import pandas as pd
import numpy as np
import os
import argparse
from prediction.data_loader import load_traffic_data
from prediction.graph_builder import build_graphs
from prediction.feature_engineering import build_spatio_temporal_features
from prediction.models import train_and_predict_all
from prediction.evaluate import evaluate_predictions, log_experiment_results
from prediction.visualize import plot_performance_metrics, plot_predictions_vs_actual

def run_benchmark():
    print("=" * 60)
    print("      LUMINA SMART GRID - AI PREDICTION PIPELINE BENCHMARK")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Step 1] Loading raw datasets...")
    df_raw = load_traffic_data()
    road_g, _, _ = build_graphs()
    print(f"Loaded traffic dataset with shape: {df_raw.shape}")
    print(f"Road network graph consists of: {road_g.number_of_nodes()} nodes, {road_g.number_of_edges()} edges")
    
    # 2. Feature Engineering
    print("\n[Step 2] Engineering spatio-temporal and graph features...")
    df_features = build_spatio_temporal_features(df_raw, road_g)
    print(f"Feature matrix successfully constructed. Shape: {df_features.shape}")
    
    # 3. Train/Test Split (Chronological to prevent leakage)
    print("\n[Step 3] Splitting dataset chronologically...")
    unique_timestamps = sorted(df_features['Timestamp'].unique())
    split_idx = int(len(unique_timestamps) * 0.833) # 25 days out of 30
    split_time = unique_timestamps[split_idx]
    
    df_train = df_features[df_features['Timestamp'] < split_time].copy().reset_index(drop=True)
    df_test = df_features[df_features['Timestamp'] >= split_time].copy().reset_index(drop=True)
    
    print(f"Train period: {df_train['Timestamp'].min()} to {df_train['Timestamp'].max()} ({df_train.shape[0]} samples)")
    print(f"Test period: {df_test['Timestamp'].min()} to {df_test['Timestamp'].max()} ({df_test.shape[0]} samples)")
    
    targets = {
        'Pedestrians': 'Target_Pedestrians',
        'Vehicles': 'Target_Vehicles'
    }
    
    all_metrics = {}
    
    # 4. Model Benchmarking
    for label, target_col in targets.items():
        print("\n" + "-"*50)
        print(f" TRAINING BENCHMARK FOR TARGET: {label.upper()}")
        print("-"*50)
        
        # Train and Predict
        predictions, y_true = train_and_predict_all(df_train, df_test, target_col)
        
        # Evaluate
        metrics_df = evaluate_predictions(predictions, y_true)
        all_metrics[label] = metrics_df
        
        print(f"\nBenchmark Results for {label}:")
        print(metrics_df.to_markdown(index=False))
        
        # Log to Excel
        log_experiment_results(metrics_df, label)
        
        # 5. Visualizations
        print(f"\n[Step 5] Generating visualization plots for {label}...")
        plot_performance_metrics(metrics_df, label)
        
        # Plot predictions vs actual for top degree node (e.g., Node_001)
        plot_predictions_vs_actual(df_test, predictions, target_col, 'Node_001', num_slots=96) # 48 hours
        
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("Generated charts and forecasts can be viewed in scratch/plots/")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
