import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import json
from xgboost import XGBClassifier

class ModelComparison:
    def __init__(self):
        with open('model_params.json', 'r') as f:
            params = json.load(f)

        # Update models to use classifiers instead of regressors
        self.models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': params['random_forest']
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': params['gradient_boosting']
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42),
                'params': params['logistic']
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42),
                'params': params['xgboost']
            }
        }
        self.results = {}

    def evaluate_models(self, X_train, y_train, X_val, y_val):
        """
        Train and evaluate multiple models with grid search using classification metrics
        """
        self.imputer = SimpleImputer(strategy='mean')
        
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_val_imputed = self.imputer.transform(X_val)
        
        evaluation_results = []
        
        for model_name, model_info in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            if model_info['params']:
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=5,
                    scoring='roc_auc',  # Changed scoring metric
                    n_jobs=-1
                )
                grid_search.fit(X_train_imputed, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                best_model = model_info['model']
                best_model.fit(X_train_imputed, y_train)
                best_params = {}

            # Make predictions
            train_pred = best_model.predict(X_train_imputed)
            train_pred_proba = best_model.predict_proba(X_train_imputed)[:, 1]
            
            val_pred = best_model.predict(X_val_imputed)
            val_pred_proba = best_model.predict_proba(X_val_imputed)[:, 1]

            # Calculate classification metrics
            results = {
                'Model': model_name,
                'Best Parameters': best_params,
                'Train ROC-AUC': roc_auc_score(y_train, train_pred_proba),
                'Val ROC-AUC': roc_auc_score(y_val, val_pred_proba),
                'Train Accuracy': accuracy_score(y_train, train_pred),
                'Val Accuracy': accuracy_score(y_val, val_pred),
                'Train Precision': precision_score(y_train, train_pred, average='weighted'),
                'Val Precision': precision_score(y_val, val_pred, average='weighted'),
                'Train Recall': recall_score(y_train, train_pred, average='weighted'),
                'Val Recall': recall_score(y_val, val_pred, average='weighted'),
                'Train F1': f1_score(y_train, train_pred, average='weighted'),
                'Val F1': f1_score(y_val, val_pred, average='weighted'),
                'Train Macro F1': f1_score(y_train, train_pred, average='macro'),
                'Val Macro F1': f1_score(y_val, val_pred, average='macro'),
                'Train Micro F1': f1_score(y_train, train_pred, average='micro'),
                'Val Micro F1': f1_score(y_val, val_pred, average='micro')
            }
            
            evaluation_results.append(results)
        
        self.results = pd.DataFrame(evaluation_results)
        return self.results

    def plot_results(self):
        """
        Visualize model comparison results with line charts
        """
        if self.results.empty:
            print("No results to plot. Run evaluate_models first.")
            return

        sns.set_theme()
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)

        # Prepare data for plotting
        metrics = {
            'ROC-AUC': ['Train ROC-AUC', 'Val ROC-AUC'],
            'Accuracy': ['Train Accuracy', 'Val Accuracy'],
            'Precision-Recall': ['Val Precision', 'Val Recall'],
            'F1 Scores': ['Val F1', 'Val Macro F1', 'Val Micro F1']
        }

        # Plot each metric
        for (metric_name, columns), ax in zip(metrics.items(), axes.flatten()):
            df_plot = self.results.set_index('Model')[columns]
            df_plot.plot(ax=ax, marker='o', linewidth=2, markersize=8)
            
            ax.set_title(f'{metric_name} Comparison', pad=15)
            ax.set_xlabel('Models')
            ax.set_ylabel(metric_name)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on points
            for line in ax.lines:
                for x, y in zip(range(len(line.get_xdata())), line.get_ydata()):
                    ax.annotate(f'{y:.3f}', 
                              (x, y),
                              textcoords="offset points",
                              xytext=(0,10),
                              ha='center')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

        # Create a separate figure for training vs validation comparison
        plt.figure(figsize=(12, 6))
        
        # Prepare data for training vs validation comparison
        train_val_metrics = {
            'ROC-AUC': ['Train ROC-AUC', 'Val ROC-AUC'],
            'Accuracy': ['Train Accuracy', 'Val Accuracy'],
            'F1': ['Train F1', 'Val F1']
        }

        for i, (metric_name, columns) in enumerate(train_val_metrics.items()):
            plt.subplot(1, 3, i+1)
            df_plot = self.results.set_index('Model')[columns]
            df_plot.plot(marker='o', linewidth=2, markersize=8)
            
            plt.title(f'{metric_name}: Training vs Validation')
            plt.xlabel('Models')
            plt.ylabel(metric_name)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def save_results(self, filename='model_comparison_results.csv'):
        """
        Save comparison results to CSV
        """
        if not self.results.empty:
            self.results.to_csv(filename, index=False)
            print(f"Results saved to {filename}")

    def select_best_model(self):
        """Select the best model based on ROC-AUC score"""
        # Use 'Val ROC-AUC' instead of 'ROC-AUC'
        best_model = self.results.loc[self.results['Val ROC-AUC'].idxmax()]
        
        print("\nBest Model Selected:")
        print(f"Model: {best_model['Model']}")
        print(f"ROC-AUC Score: {best_model['Val ROC-AUC']:.3f}")
        print(f"Accuracy: {best_model['Val Accuracy']:.3f}")
        print(f"Precision: {best_model['Val Precision']:.3f}")
        print(f"Recall: {best_model['Val Recall']:.3f}")
        print(f"F1 Score: {best_model['Val F1']:.3f}")
        
        return best_model

    def preprocess_data(self, X):
        """Helper method to preprocess data using the fitted imputer"""
        if not hasattr(self, 'imputer'):
            raise ValueError("Imputer not initialized. Run evaluate_models first.")
        return self.imputer.transform(X)