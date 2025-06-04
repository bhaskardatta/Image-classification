"""
Main script to run the complete CIFAR-10 image classification pipeline
"""
import os
import sys
sys.path.append('src')
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import analyze_model_performance, create_analysis_report


def main():
    """Run the complete pipeline"""
    print("CIFAR-10 Image Classification from Scratch")
    print("=" * 50)
    print("This script will:")
    print("1. Download and preprocess CIFAR-10 data")
    print("2. Train a neural network from scratch")
    print("3. Evaluate the model performance")
    print("4. Generate analysis report")
    print()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Train the model
    print("Step 1: Training the neural network...")
    print("-" * 30)
    try:
        model, history = train_model()
        print("✓ Training completed successfully!")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    print()
    
    # Step 2: Evaluate the model
    print("Step 2: Evaluating the model...")
    print("-" * 30)
    try:
        eval_results = evaluate_model()
        print("✓ Evaluation completed successfully!")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return
    
    print()
    
    # Step 3: Generate analysis
    print("Step 3: Generating performance analysis...")
    print("-" * 30)
    try:
        analyze_model_performance()
        
        # Save analysis report
        analysis_text = create_analysis_report()
        with open('results/analysis_report.txt', 'w') as f:
            f.write(analysis_text)
        
        print("✓ Analysis completed successfully!")
        print()
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return
    
    # Summary
    print("Pipeline completed successfully!")
    print()
    print("Generated files in 'results/' directory:")
    print("- trained_model.pkl          (Trained neural network)")
    print("- training_history.pkl       (Training metrics)")
    print("- training_curves.png        (Loss and accuracy plots)")
    print("- evaluation_report.pkl      (Evaluation metrics)")
    print("- evaluation_report.txt      (Text summary)")
    print("- confusion_matrix.png       (Confusion matrix plot)")
    print("- analysis_report.txt        (200-word analysis)")
    print()
    
    # Display final results
    final_accuracy = eval_results['accuracy']
    target_met = "✓ YES" if final_accuracy >= 0.6 else "✗ NO"
    
    print("Final Results Summary:")
    print(f"- Test Accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
    print(f"- Target 60% met: {target_met}")
    print(f"- Macro F1-score: {eval_results['macro_f1']:.3f}")
    print(f"- Classes: {', '.join(eval_results['class_names'])}")


if __name__ == "__main__":
    main()
