import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import config as cfg


class ThresholdOptimizer:
    """Optimize decision thresholds for ordinal classification"""
    
    def __init__(self, y_true, y_probs):
        """
        Args:
            y_true: true labels [N]
            y_probs: predicted probabilities [N, num_classes]
        """
        self.y_true = y_true
        self.y_probs = y_probs
        self.num_classes = y_probs.shape[1]
        self.best_thresholds = None
        self.best_kappa = -1
    
    def ordinal_prediction(self, probs, thresholds):
        """
        Convert probabilities to ordinal predictions using thresholds
        
        For DR: 0 (No) -> 1 (Mild) -> 2 (Mod) -> 3 (Sev) -> 4 (Prolif)
        Use cumulative probabilities
        """
        # Cumulative probabilities
        cum_probs = np.cumsum(probs, axis=1)
        
        # Default thresholds if none provided
        if thresholds is None:
            thresholds = [0.5, 0.5, 0.5, 0.5]
        
        preds = np.zeros(len(probs), dtype=int)
        
        for i in range(len(probs)):
            pred = 0
            for t_idx, threshold in enumerate(thresholds):
                if cum_probs[i, t_idx] < threshold:
                    pred = t_idx + 1
                else:
                    break
            preds[i] = min(pred, self.num_classes - 1)
        
        return preds
    
    def optimize_thresholds_grid_search(self, n_steps=20):
        """
        Grid search for optimal thresholds
        
        Search space: each threshold in [0.3, 0.7]
        """
        print("Optimizing thresholds via grid search...")
        
        threshold_range = np.linspace(0.3, 0.7, n_steps)
        best_kappa = -1
        best_thresholds = [0.5] * (self.num_classes - 1)
        
        # Grid search (simplified - full grid too expensive)
        # Search each threshold independently
        for t_idx in range(self.num_classes - 1):
            best_t = 0.5
            
            for t_val in tqdm(threshold_range, desc=f"Optimizing threshold {t_idx}"):
                test_thresholds = best_thresholds.copy()
                test_thresholds[t_idx] = t_val
                
                preds = self.ordinal_prediction(self.y_probs, test_thresholds)
                kappa = cohen_kappa_score(self.y_true, preds, weights='quadratic')
                
                if kappa > best_kappa:
                    best_kappa = kappa
                    best_t = t_val
            
            best_thresholds[t_idx] = best_t
            print(f"  Threshold {t_idx}: {best_t:.3f} (Kappa: {best_kappa:.4f})")
        
        self.best_thresholds = best_thresholds
        self.best_kappa = best_kappa
        
        return best_thresholds, best_kappa
    
    def optimize_thresholds_class_specific(self):
        """
        Optimize thresholds by maximizing per-class F1 scores
        Works better for imbalanced datasets
        """
        print("Optimizing class-specific thresholds...")
        
        best_thresholds = []
        
        for class_idx in range(self.num_classes):
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.linspace(0.1, 0.9, 50):
                # Binary classification: class vs rest
                binary_preds = (self.y_probs[:, class_idx] >= threshold).astype(int)
                binary_true = (self.y_true == class_idx).astype(int)
                
                # Calculate F1
                tp = np.sum((binary_preds == 1) & (binary_true == 1))
                fp = np.sum((binary_preds == 1) & (binary_true == 0))
                fn = np.sum((binary_preds == 0) & (binary_true == 1))
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            best_thresholds.append(best_threshold)
            print(f"  Class {class_idx}: threshold={best_threshold:.3f}, F1={best_f1:.3f}")
        
        # Convert to predictions
        preds = np.argmax(self.y_probs, axis=1)
        
        # Apply thresholds
        for i in range(len(preds)):
            class_probs_above_threshold = [
                self.y_probs[i, c] >= best_thresholds[c] 
                for c in range(self.num_classes)
            ]
            
            if any(class_probs_above_threshold):
                # Choose highest prob among those above threshold
                valid_classes = [c for c in range(self.num_classes) if class_probs_above_threshold[c]]
                preds[i] = max(valid_classes, key=lambda c: self.y_probs[i, c])
        
        kappa = cohen_kappa_score(self.y_true, preds, weights='quadratic')
        
        self.best_thresholds = best_thresholds
        self.best_kappa = kappa
        
        return best_thresholds, kappa
    
    def predict_with_thresholds(self, probs, thresholds=None):
        """Apply optimized thresholds to new probabilities"""
        if thresholds is None:
            thresholds = self.best_thresholds
        
        if thresholds is None:
            # Default: argmax
            return np.argmax(probs, axis=1)
        
        # Apply class-specific thresholds
        preds = np.argmax(probs, axis=1)
        
        for i in range(len(probs)):
            class_probs_above_threshold = [
                probs[i, c] >= thresholds[c] 
                for c in range(len(thresholds))
            ]
            
            if any(class_probs_above_threshold):
                valid_classes = [c for c in range(len(thresholds)) if class_probs_above_threshold[c]]
                preds[i] = max(valid_classes, key=lambda c: probs[i, c])
        
        return preds
    
    def plot_threshold_impact(self, save_path=None):
        """Visualize how thresholds affect predictions"""
        if self.best_thresholds is None:
            print("Run optimization first!")
            return
        
        # Compare default vs optimized
        default_preds = np.argmax(self.y_probs, axis=1)
        optimized_preds = self.predict_with_thresholds(self.y_probs)
        
        default_kappa = cohen_kappa_score(self.y_true, default_preds, weights='quadratic')
        optimized_kappa = cohen_kappa_score(self.y_true, optimized_preds, weights='quadratic')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion matrices
        cm_default = confusion_matrix(self.y_true, default_preds)
        cm_optimized = confusion_matrix(self.y_true, optimized_preds)
        
        labels = ["No DR", "Mild", "Mod", "Sev", "Prolif"]
        
        im1 = axes[0].imshow(cm_default, cmap='Blues')
        axes[0].set_title(f"Default (Kappa: {default_kappa:.4f})")
        axes[0].set_xticks(range(5))
        axes[0].set_yticks(range(5))
        axes[0].set_xticklabels(labels, rotation=45)
        axes[0].set_yticklabels(labels)
        
        im2 = axes[1].imshow(cm_optimized, cmap='Greens')
        axes[1].set_title(f"Optimized (Kappa: {optimized_kappa:.4f})")
        axes[1].set_xticks(range(5))
        axes[1].set_yticks(range(5))
        axes[1].set_xticklabels(labels, rotation=45)
        axes[1].set_yticklabels(labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
        print(f"\nThreshold Impact:")
        print(f"  Default Kappa: {default_kappa:.4f}")
        print(f"  Optimized Kappa: {optimized_kappa:.4f}")
        print(f"  Improvement: +{(optimized_kappa - default_kappa):.4f}")