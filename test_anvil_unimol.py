#!/usr/bin/env python3
"""
Test script to demonstrate anvil-like workflow with UniMol
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from openadmet.models.architecture.unimol import UniMolModel

def main():
    print("Testing anvil-like workflow with UniMol")
    print("=" * 60)
    
    print("\n1. Loading data...")
    data_path = "openadmet/models/tests/unit/test_data/CYP3A4_chembl_pchembl.csv"
    data = pd.read_csv(data_path)
    
    # Prepare data for UniMol (using regression instead of classification)
    # Since all compounds are active, we'll predict pChEMBL values
    data_processed = data[['CANONICAL_SMILES', 'pChEMBL mean']].copy()
    data_processed.columns = ['SMILES', 'TARGET']
    
    # Use only first 100 compounds for demo
    data_processed = data_processed.head(100)
    
    print(f"Loaded {len(data_processed)} samples")
    print("Data preview:")
    print(data_processed.head())
    
    print("\n2. Splitting data...")
    train_size = int(0.8 * len(data_processed))
    val_size = int(0.1 * len(data_processed))
    
    train_data = data_processed[:train_size]
    val_data = data_processed[train_size:train_size + val_size]
    test_data = data_processed[train_size + val_size:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    print("\n3. Creating UniMol model...")
    model = UniMolModel(
        task="regression",        # Changed to regression for pChEMBL prediction
        epochs=3,  
        batch_size=8,
        metrics="mse",           # Changed to MSE for regression
        model_name="unimolv1",
        remove_hs=False
    )
    
    print("Model configuration:")
    for key, value in model.get_model_summary().items():
        print(f"  {key}: {value}")
    
    print("\n4. Training model...")
    model.build()
    
    try:
        model.train(train_data)
        print("Training completed successfully!")
        
        print("\n5. Making predictions on test data...")
        predictions = model.predict(test_data)
        
        print("Prediction results:")
        for i, (_, row) in enumerate(test_data.iterrows()):
            pred_value = predictions[i][0] if predictions.ndim > 1 else predictions[i]
            actual_value = row['TARGET']
            error = abs(pred_value - actual_value)
            print(f"  {row['SMILES'][:30]:<30}: Pred={pred_value:.4f}, Actual={actual_value:.4f}, Error={error:.4f}")
        
        print("\n6. Model evaluation:")
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        actual = test_data['TARGET'].values
        pred_values = predictions.flatten()
        
        mse = mean_squared_error(actual, pred_values)
        mae = mean_absolute_error(actual, pred_values)
        r2 = 1 - mse / np.var(actual) if np.var(actual) > 0 else 0
        
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        print("\nAnvil-like workflow completed successfully!")
        print("\nTo use with actual anvil command:")
        print("  openadmet anvil --recipe-path openadmet/models/tests/unit/test_data/basic_anvil_unimol.yaml")
        
    except Exception as e:
        print(f"Error during workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
