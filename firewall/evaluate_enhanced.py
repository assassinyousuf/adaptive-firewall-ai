"""
Enhanced Model Evaluation Script

This script evaluates the enhanced DQN model on the held-out test set.
It provides comprehensive metrics including confusion matrix, precision,
recall, F1 score, and per-class performance.
"""

import numpy as np
from stable_baselines3 import DQN
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from firewall.env import FirewallEnv
from firewall.features import extract_features


def evaluate_model(model_path, test_data_path):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test dataset (.npy format)
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*60)
    print("ENHANCED MODEL EVALUATION")
    print("="*60)
    
    # Load model
    print(f"\n[1/5] Loading model from {model_path}...")
    try:
        model = DQN.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load test data
    print(f"\n[2/5] Loading test data from {test_data_path}...")
    try:
        test_data = np.load(test_data_path)
        print(f"✅ Loaded {len(test_data)} test samples")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None
    
    # Create environment for evaluation
    print("\n[3/5] Creating evaluation environment...")
    env = FirewallEnv(test_data)
    
    # Run evaluation
    print("\n[4/5] Running model predictions...")
    y_true = []
    y_pred = []
    
    obs, _ = env.reset()
    for i in range(len(test_data)):
        # Get model prediction
        action, _ = model.predict(obs, deterministic=True)
        
        # Store true label and prediction
        true_label = int(test_data[i, -1])  # Last column is label
        y_true.append(true_label)
        y_pred.append(int(action))
        
        # Step environment
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
    
    # Calculate metrics
    print("\n[5/5] Calculating metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n📊 Overall Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    
    print("\n🎯 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                Benign  Malicious")
    print(f"  Actual Benign    {tn:4d}     {fp:4d}")
    print(f"         Malicious {fn:4d}     {tp:4d}")
    
    print("\n📈 Detailed Metrics:")
    print(f"  True Negatives:  {tn} (correctly identified benign)")
    print(f"  True Positives:  {tp} (correctly identified malicious)")
    print(f"  False Positives: {fp} (benign flagged as malicious)")
    print(f"  False Negatives: {fn} (malicious missed)")
    
    # False positive/negative rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n⚠️ Error Rates:")
    print(f"  False Positive Rate: {fpr*100:.2f}% (benign blocked)")
    print(f"  False Negative Rate: {fnr*100:.2f}% (attacks missed)")
    
    print("\n📋 Per-Class Performance:")
    print("  Class     | Precision | Recall | F1-Score | Support")
    print("  " + "-"*56)
    print(f"  Benign    |   {precision_per_class[0]*100:5.2f}% | {recall_per_class[0]*100:5.2f}% |  {f1_per_class[0]*100:5.2f}%  | {support_per_class[0]:4d}")
    print(f"  Malicious |   {precision_per_class[1]*100:5.2f}% | {recall_per_class[1]*100:5.2f}% |  {f1_per_class[1]*100:5.2f}%  | {support_per_class[1]:4d}")
    
    # Classification report
    print("\n📄 Detailed Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Benign', 'Malicious'],
        digits=4
    ))
    
    # Performance assessment
    print("\n" + "="*60)
    print("PERFORMANCE ASSESSMENT")
    print("="*60)
    
    if accuracy >= 0.95:
        print("✅ EXCELLENT: Model performance is production-ready")
    elif accuracy >= 0.90:
        print("✅ GOOD: Model performance is acceptable for deployment")
    elif accuracy >= 0.85:
        print("⚠️ FAIR: Model needs improvement before production use")
    else:
        print("❌ POOR: Model requires significant improvement")
    
    if fpr <= 0.05:
        print("✅ LOW FALSE POSITIVES: Minimal disruption to legitimate traffic")
    elif fpr <= 0.10:
        print("⚠️ MODERATE FALSE POSITIVES: Some legitimate traffic may be blocked")
    else:
        print("❌ HIGH FALSE POSITIVES: Too many false alarms")
    
    if fnr <= 0.05:
        print("✅ LOW FALSE NEGATIVES: Strong attack detection")
    elif fnr <= 0.10:
        print("⚠️ MODERATE FALSE NEGATIVES: Some attacks may slip through")
    else:
        print("❌ HIGH FALSE NEGATIVES: Missing too many attacks")
    
    print("\n" + "="*60)
    
    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }


def compare_with_baseline():
    """Compare enhanced model with original model."""
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)
    
    print("\n📊 Model Comparison:")
    print("\n  Metric          | Original | Enhanced | Improvement")
    print("  " + "-"*55)
    print("  Features        |    3     |    7     |   +133%")
    print("  Dataset Size    |   100    |  5,000   |  +4900%")
    print("  Network Size    |  64x64   | 128x128  |   +300%")
    print("  Buffer Size     |  10,000  | 100,000  |   +900%")
    print("  Validation Set  |    No    |   Yes    |     ✅")
    print("  Early Stopping  |    No    |   Yes    |     ✅")
    
    print("\n💡 Key Improvements:")
    print("  ✅ Advanced feature engineering (entropy, variance, timing)")
    print("  ✅ Larger and more diverse training dataset")
    print("  ✅ Professional training pipeline with validation")
    print("  ✅ Better hyperparameters and larger network")
    print("  ✅ Reduced overfitting risk with early stopping")
    print("  ✅ More realistic attack detection capabilities")


if __name__ == "__main__":
    # Paths
    model_path = "model/firewall_dqn_enhanced.zip"
    test_data_path = "model/test_data.npy"
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please run training first: python -m firewall.train_enhanced")
        sys.exit(1)
    
    if not Path(test_data_path).exists():
        print(f"❌ Error: Test data not found at {test_data_path}")
        print("Please run training first: python -m firewall.train_enhanced")
        sys.exit(1)
    
    # Run evaluation
    results = evaluate_model(model_path, test_data_path)
    
    if results:
        # Show comparison
        compare_with_baseline()
        
        print("\n✅ Evaluation complete!")
        print("\n💾 Model ready for deployment at:", model_path)
        print("\n🚀 Next steps:")
        print("  1. Review metrics above")
        print("  2. If satisfied, test with: python demo.py")
        print("  3. Deploy in observe mode: python -m runtime.sniff --observe")
        print("  4. When ready, activate: python -m runtime.sniff --active")
