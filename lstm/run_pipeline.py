"""
Main Pipeline Script - Run Complete LSTM Training Workflow
Execute this script to run the entire pipeline from exploration to training
"""

import os
import sys
import subprocess


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print_section(description)
    print(f"Running: {script_path}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n✗ Error running {script_path}")
        print("Please check the error messages above")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def main():
    """Main pipeline execution"""
    print("=" * 70)
    print("LSTM STOCK PRICE PREDICTION - COMPLETE PIPELINE")
    print("=" * 70)
    print("\nThis script will execute the following steps:")
    print("1. Data Exploration")
    print("2. Data Preprocessing") 
    print("3. Dataset Creation Test")
    print("4. Model Architecture Test")
    print("5. Model Training")
    print("\n" + "=" * 70)
    
    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline cancelled")
        return
    
    # Step 1: Data Exploration
    if not run_script("src/data/explore.py", "STEP 1: DATA EXPLORATION"):
        return
    
    input("\nPress Enter to continue to preprocessing...")
    
    # Step 2: Data Preprocessing
    if not run_script("src/data/preprocessor.py", "STEP 2: DATA PREPROCESSING"):
        return
    
    input("\nPress Enter to continue to dataset test...")
    
    # Step 3: Dataset Test
    if not run_script("src/data/dataset.py", "STEP 3: DATASET CREATION TEST"):
        return
    
    input("\nPress Enter to continue to model test...")
    
    # Step 4: Model Test
    if not run_script("src/models/lstm.py", "STEP 4: MODEL ARCHITECTURE TEST"):
        return
    
    input("\nPress Enter to start training...")
    
    # Step 5: Training
    if not run_script("src/training/trainer.py", "STEP 5: MODEL TRAINING"):
        return
    
    # Summary
    print_section("PIPELINE COMPLETE")
    print("\nAll steps completed successfully!")
    print("\nGenerated outputs:")
    print("  • data/exploration_results/ - Visualizations")
    print("  • data/processed/NABIL_processed.csv - Processed data")
    print("  • models/scalers/ - Fitted scalers")
    print("  • models/checkpoints/best_model.pth - Best model")
    print("  • models/lstm_final.pth - Final model")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review training results")
    print("2. Create evaluation script for metrics")
    print("3. Build inference script for predictions")
    print("4. Experiment with different architectures")
    print("=" * 70)


if __name__ == "__main__":
    main()
