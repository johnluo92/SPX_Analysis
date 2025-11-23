# core/feature_correlation_analyzer.py
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class FeatureCorrelationAnalyzer:
    def __init__(self, threshold=0.85, min_importance=0.001):
        self.threshold = threshold
        self.min_importance = min_importance
        self.correlation_matrix = None
        self.removed_features = []
        self.kept_features = []

    def analyze_and_remove(self, features_df, importance_scores, protected_features):
        """
        Main method: removes correlated features based on importance
        """
        # 1. Compute correlation matrix
        self.correlation_matrix = features_df.corr().abs()

        # 2. Find correlated pairs
        upper_tri = np.triu(np.ones_like(self.correlation_matrix), k=1)
        high_corr_pairs = []

        for i in range(len(self.correlation_matrix)):
            for j in range(i+1, len(self.correlation_matrix)):
                if self.correlation_matrix.iloc[i, j] > self.threshold:
                    feat_i = self.correlation_matrix.index[i]
                    feat_j = self.correlation_matrix.columns[j]
                    corr_val = self.correlation_matrix.iloc[i, j]
                    high_corr_pairs.append((feat_i, feat_j, corr_val))

        # 3. Decide which to keep based on importance
        to_remove = set()
        for feat_i, feat_j, corr in high_corr_pairs:
            if feat_i in protected_features and feat_j in protected_features:
                continue  # Keep both if both protected
            elif feat_i in protected_features:
                to_remove.add(feat_j)
            elif feat_j in protected_features:
                to_remove.add(feat_i)
            else:
                # Remove lower importance feature
                imp_i = importance_scores.get(feat_i, 0)
                imp_j = importance_scores.get(feat_j, 0)
                if imp_i < imp_j:
                    to_remove.add(feat_i)
                else:
                    to_remove.add(feat_j)

        self.removed_features = list(to_remove)
        self.kept_features = [f for f in features_df.columns if f not in to_remove]

        return self.kept_features, self.removed_features

    def generate_report(self, output_dir="diagnostics", suffix=""):
        """Generate correlation heatmap and report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save correlation pairs
        report = {
            "threshold": self.threshold,
            "total_features_before": len(self.correlation_matrix),
            "features_removed": len(self.removed_features),
            "features_kept": len(self.kept_features),
            "removed_features": self.removed_features
        }

        report_filename = f"correlation_report{suffix}.json"
        with open(output_path / report_filename, "w") as f:
            json.dump(report, f, indent=2)

        # Generate heatmap (top 50 features only)
        top_50 = self.kept_features[:50] if len(self.kept_features) > 50 else self.kept_features
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            self.correlation_matrix.loc[top_50, top_50],
            cmap="RdYlBu_r",
            center=0,
            square=True,
            linewidths=0.5
        )
        title_suffix = suffix.replace("_", " ").title() if suffix else ""
        plt.title(f"Feature Correlation Matrix{title_suffix} (Top 50 Features)")
        plt.tight_layout()
        heatmap_filename = f"correlation_heatmap{suffix}.png"
        plt.savefig(output_path / heatmap_filename, dpi=150)
        plt.close()
