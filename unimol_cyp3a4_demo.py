#!/usr/bin/env python3
"""
UniMol Model Demo Script with CYP3A4 Data

Demonstrates how to use UniMol models for molecular property prediction
using the CYP3A4 ChEMBL dataset.
"""

import pandas as pd
import numpy as np
import tempfile
import os

# Add project path
import sys
sys.path.insert(0, '.')

from openadmet.models.architecture.unimol import UniMolModel

def main():
    print("UniMol Model Demo with CYP3A4 Data")
    print("=" * 50)
    
    # 1. Load CYP3A4 data
    print("\n1. Loading CYP3A4 ChEMBL data...")
    data_path = "openadmet/models/tests/unit/test_data/CYP3A4_chembl_pchembl.csv"
    
    try:
        full_data = pd.read_csv(data_path)
        print(f"Loaded {len(full_data)} molecules from CYP3A4 dataset")
        print(f"Columns: {list(full_data.columns)}")
        
        # Use the first 100 molecules for demo
        data_sample = full_data.head(100).copy()
        print(f"Using the first {len(data_sample)} molecules for demo")
        
        # Prepare data for UniMol
        # Use CANONICAL_SMILES as input and 'pChEMBL mean' as regression target
        train_data = data_sample[['CANONICAL_SMILES', 'pChEMBL mean']].copy()
        train_data.columns = ['SMILES', 'TARGET']  # Rename to match UniMol expected format
        
        print(f"Target statistics:")
        print(f"  Mean pChEMBL: {train_data['TARGET'].mean():.3f}")
        print(f"  Std pChEMBL: {train_data['TARGET'].std():.3f}")
        print(f"  Range: {train_data['TARGET'].min():.3f} - {train_data['TARGET'].max():.3f}")
        
        # Split into train/test (80/20 split)
        train_size = int(0.8 * len(train_data))
        train_subset = train_data[:train_size]
        test_subset = train_data[train_size:].drop('TARGET', axis=1)  # Remove target for prediction
        test_targets = train_data[train_size:]['TARGET']  # Keep targets for evaluation
        
        print(f"Training set: {len(train_subset)} molecules")
        print(f"Test set: {len(test_subset)} molecules")
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at {data_path}")
        print("Please make sure the file exists and the path is correct.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Create UniMol model for regression
    print("\n2. Creating UniMol model for pChEMBL regression...")
    model = UniMolModel(
        task='regression',        # regression task to predict pChEMBL values
        epochs=5,                 # fewer epochs for demo
        batch_size=16,           # smaller batch size for demo
        model_name='unimolv1',   # use UniMol v1
        model_size='84m',        # smallest model for faster training
        metrics='mse',           # use MSE as evaluation metric for regression
        remove_hs=False          # keep hydrogen atom information
    )
    
    print("Model parameters:")
    for key, value in model.get_model_summary().items():
        print(f"  {key}: {value}")
    
    # 3. Build model
    print("\n3. Building model...")
    model.build()
    
    # 4. Train model
    print("\n4. Starting model training...")
    print("Note: This may take several minutes, especially on first run when downloading pre-trained models...")
    print("Training with CYP3A4 pChEMBL data (predicting binding affinity values)...")
    
    try:
        # Save train data to CSV for UniMol
        train_csv = "temp_train_data.csv"
        train_subset.to_csv(train_csv, index=False)
        print(f"Training data saved to temporary file: {train_csv}")
        
        model.train(train_csv)
        print("Model training completed!")
        
        # 5. Make predictions
        print("\n5. Making predictions on test set...")
        test_csv = "temp_test_data.csv"
        test_subset.to_csv(test_csv, index=False)
        
        predictions = model.predict(test_csv)
        
        print("Prediction results (first 10 compounds):")
        for i, (smiles, pred, actual) in enumerate(zip(test_subset['SMILES'][:10], predictions.flatten()[:10], test_targets[:10])):
            # For regression, pred is the predicted pChEMBL value
            print(f"  {smiles[:30]:<30}: Predicted={pred:.4f}, Actual={actual:.4f}, Error={abs(pred-actual):.4f}")
        
        # Calculate performance metrics
        mse = np.mean((predictions.flatten() - test_targets) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - test_targets))
        r2 = 1 - mse / np.var(test_targets)
        
        print(f"\nModel Performance on Test Set ({len(test_subset)} molecules):")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        print(f"\nPrediction statistics:")
        print(f"  Mean predicted pChEMBL: {np.mean(predictions):.4f}")
        print(f"  Std predicted pChEMBL: {np.std(predictions):.4f}")
        print(f"  Range: {np.min(predictions):.4f} - {np.max(predictions):.4f}")
        
        print(f"\nActual target statistics:")
        print(f"  Mean actual pChEMBL: {np.mean(test_targets):.4f}")
        print(f"  Std actual pChEMBL: {np.std(test_targets):.4f}")
        print(f"  Range: {np.min(test_targets):.4f} - {np.max(test_targets):.4f}")
        
        # 6. Save model
        print("\n6. Saving model...")
        model_path = "unimol_cyp3a4_model.pkl"
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Clean up temporary files
        for temp_file in [train_csv, test_csv, model_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print("Temporary files cleaned up")
        
    except Exception as e:
        print(f"Error during training/prediction: {e}")
        print("This could be due to:")
        print("1. Missing unimol_tools package")
        print("2. Network connection issues (unable to download pre-trained models)") 
        print("3. Insufficient memory")
        print("4. GPU/CUDA issues")
        
        # Clean up on error
        for temp_file in ["temp_train_data.csv", "temp_test_data.csv"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return
    
    print("\nDemo completed successfully!")
    print("\nTo use UniMol with the full dataset via anvil:")
    print("   openadmet anvil --recipe-path openadmet/models/tests/unit/test_data/basic_anvil_unimol.yaml")

if __name__ == "__main__":
    main()
