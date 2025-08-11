#!/usr/bin/env python3
"""
UniMol Model Demo Script

Demonstrates how to use UniMol models for molecular property prediction
within the OpenADMET framework.
"""

import pandas as pd
import tempfile
import os

# Add project path
import sys
sys.path.insert(0, '.')

from openadmet.models.architecture.unimol import UniMolModel

def main():
    print("UniMol Model Demo")
    print("=" * 50)

    # 1. Create sample data
    print("\n1. Creating sample training data...")
    train_data = pd.DataFrame({
        'SMILES': [  # Use uppercase SMILES as expected by UniMol
            'CCO',           # ethanol
            'CCC',           # propane
            'CCCC',          # butane
            'CCCCC',         # pentane
            'CC(C)C',        # isobutane
            'CC(C)CC',       # isopentane
            'CCCCCC',        # hexane
            'CC(C)CCC',      # 2-methylpentane
        ],
        'TARGET': [0.1, 0.2, 0.3, 0.4, 0.25, 0.35, 0.5, 0.4]  # simulated molecular property values
    })

    test_data = pd.DataFrame({
        'SMILES': [  # Use uppercase SMILES as expected by UniMol
            'CCCCCCC',       # heptane
            'CC(C)CCCC',     # 2-methylhexane
        ]
    })

    print(f"Training data: {len(train_data)} molecules")
    print(f"Test data: {len(test_data)} molecules")

    # 2. Create UniMol model
    print("\n2. Creating UniMol model...")
    # Use more data for demo to avoid cross-validation issues
    train_data = pd.concat([train_data] * 2, ignore_index=True)  # Double the data
    print(f"Expanded training data: {len(train_data)} molecules")

    model = UniMolModel(
        task='regression',        # regression task
        epochs=2,                 # quick demo, only 2 epochs
        batch_size=4,            # small batch size
        model_name='unimolv1',   # use UniMol v1
        metrics='mse',           # use MSE as evaluation metric
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

    try:
        model.train(train_data)
        print("Model training completed!")

        # 5. Make predictions
        print("\n5. Making predictions...")
        predictions = model.predict(test_data)

        print("Prediction results:")
        for i, (smiles, pred) in enumerate(zip(test_data['SMILES'], predictions.flatten())):
            print(f"  {smiles}: {pred:.4f}")

        # 6. Save model
        print("\n6. Saving model...")
        model_path = "unimol_demo_model.pkl"
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        # 7. Test model loading
        print("\n7. Testing model loading...")
        new_model = UniMolModel()
        new_model.load(model_path)
        print("Model loaded successfully!")

        # Clean up temporary files
        if os.path.exists(model_path):
            os.remove(model_path)
            print("Temporary files cleaned up")

    except Exception as e:
        print(f"Error during training: {e}")
        print("This could be due to:")
        print("1. Missing GPU support (UniMol runs slowly on CPU)")
        print("2. Network connection issues (unable to download pre-trained models)")
        print("3. Insufficient memory")
        return

    print("\nDemo completed!")
    print("\nUsage methods:")
    print("1. Via anvil command line:")
    print("   openadmet anvil --recipe-path openadmet/models/tests/unit/test_data/basic_anvil_unimol.yaml")
    print("2. Direct use of UniMolModel class in Python")

if __name__ == "__main__":
    main()
