# Threshold Optimization for Medical Use
# Add this code to your notebook after the model comparison section

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, auc, confusion_matrix

def optimize_threshold_for_medical_use(model, X_test, y_test, model_name="Model"):
    """
    Optimize decision thresholds for medical use with comprehensive analysis.
    
    Parameters:
    model: Trained model with predict_proba method
    X_test: Test features
    y_test: Test labels
    model_name: Name of the model for display
    
    Returns:
    dict: Optimal threshold and performance metrics
    """
    
    print(f"🎯 Threshold Optimization for {model_name}")
    print("=" * 50)
    
    # Get predicted probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    
    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    
    # Compute F1 for each threshold
    f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds_pr]
    
    # Find best thresholds
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thresh = thresholds_pr[best_f1_idx]
    
    # Calculate metrics at best F1 threshold
    y_pred_best = (y_proba >= best_f1_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    
    best_metrics = {
        'threshold': best_f1_thresh,
        'precision': precision[best_f1_idx],
        'recall': recall[best_f1_idx],
        'f1_score': f1_scores[best_f1_idx],
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }
    
    print(f"Best threshold by F1-score: {best_f1_thresh:.3f}")
    print(f"Precision at best F1: {best_metrics['precision']:.3f}")
    print(f"Recall at best F1: {best_metrics['recall']:.3f}")
    print(f"F1-score: {best_metrics['f1_score']:.3f}")
    print(f"Sensitivity: {best_metrics['sensitivity']:.3f}")
    print(f"Specificity: {best_metrics['specificity']:.3f}")
    print(f"Accuracy: {best_metrics['accuracy']:.3f}")
    
    # Plot Precision-Recall Curve
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(recall, precision, label="PR Curve", linewidth=2, color='blue')
    plt.scatter(recall[best_f1_idx], precision[best_f1_idx], color="red", s=100,
                label=f"Best F1 threshold={best_f1_thresh:.2f}")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve with Optimal Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot ROC Curve
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc(fpr,tpr):.3f})", linewidth=2, color='green')
    plt.plot([0,1],[0,1],"--",color="gray", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot F1 Score vs Threshold
    plt.subplot(1, 3, 3)
    plt.plot(thresholds_pr, f1_scores, label="F1 Score", linewidth=2, color='orange')
    plt.axvline(x=best_f1_thresh, color='red', linestyle='--', alpha=0.7,
                label=f'Optimal threshold={best_f1_thresh:.2f}')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return best_metrics

def comprehensive_threshold_analysis(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive threshold analysis with detailed metrics table.
    
    Parameters:
    model: Trained model with predict_proba method
    X_test: Test features
    y_test: Test labels
    model_name: Name of the model for display
    
    Returns:
    pandas.DataFrame: Detailed metrics for different thresholds
    """
    
    print(f"📊 Comprehensive Threshold Analysis for {model_name}")
    print("=" * 60)
    
    # Get predicted probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results.append({
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Accuracy': accuracy,
            'True_Positives': tp,
            'False_Positives': fp,
            'True_Negatives': tn,
            'False_Negatives': fn
        })
    
    # Create DataFrame
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # Display results
    print("Detailed Metrics for Different Thresholds:")
    print(results_df.round(3))
    
    # Find best thresholds for different criteria
    best_f1_idx = results_df['F1_Score'].idxmax()
    best_sensitivity_idx = results_df['Sensitivity'].idxmax()
    best_balanced_idx = results_df[(results_df['Sensitivity'] > 0.7) & (results_df['Specificity'] > 0.7)]['F1_Score'].idxmax()
    
    print(f"\n🏆 Best Thresholds:")
    print(f"Best F1 Score: {results_df.loc[best_f1_idx, 'Threshold']:.2f} (F1={results_df.loc[best_f1_idx, 'F1_Score']:.3f})")
    print(f"Best Sensitivity: {results_df.loc[best_sensitivity_idx, 'Threshold']:.2f} (Sens={results_df.loc[best_sensitivity_idx, 'Sensitivity']:.3f})")
    
    if not pd.isna(best_balanced_idx):
        print(f"Best Balanced: {results_df.loc[best_balanced_idx, 'Threshold']:.2f} (F1={results_df.loc[best_balanced_idx, 'F1_Score']:.3f})")
    
    # Medical recommendation
    medical_threshold = results_df.loc[best_f1_idx, 'Threshold']
    print(f"\n💡 Medical Recommendation: Use threshold {medical_threshold:.2f}")
    print(f"This provides the best balance of precision and recall for stroke detection.")
    
    return results_df

# Example usage in your notebook:
"""
# After training your models, use this code:

# Get the best model (Logistic Regression)
best_model_name = comparison_df.iloc[0]['model']
best_model_pipeline = trained_models[best_model_name]

# Extract the classifier from pipeline
if hasattr(best_model_pipeline, 'named_steps'):
    best_model_classifier = best_model_pipeline.named_steps['classifier']
else:
    best_model_classifier = best_model_pipeline

# Optimize threshold
optimal_metrics = optimize_threshold_for_medical_use(
    best_model_pipeline, X_test_final, y_test_final, best_model_name
)

# Comprehensive analysis
threshold_analysis_df = comprehensive_threshold_analysis(
    best_model_pipeline, X_test_final, y_test_final, best_model_name
)

# Update your pipeline with optimal threshold
optimal_threshold = optimal_metrics['threshold']
improved_stroke_pipeline.optimal_threshold = optimal_threshold

print(f"✅ Pipeline updated with optimal threshold: {optimal_threshold:.3f}")
"""
