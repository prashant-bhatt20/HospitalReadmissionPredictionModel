import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import requests
import json
import pickle
import warnings
from ydata_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

warnings.filterwarnings('ignore', category=FutureWarning)

class BaseModelTrainer:
    """Base class for common model training functionality"""
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def split_data(self, X, y, test_size=0.10, val_size=0.15):
        """Updated data splitting logic with 75:15:10 ratio"""
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Common metric calculation logic"""
        return {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'micro_f1': f1_score(y_true, y_pred, average='micro')
        }

    def create_preprocessor(self, strategy):
        """Create preprocessing pipeline"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy=strategy)),
            ('scaler', StandardScaler())
        ])

class HospitalReadmissionPredictor(BaseModelTrainer):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.model_classes = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'XGBClassifier': xgb.XGBClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }

    def load_and_preprocess_data(self, api_url):
        """Load data from CMS API and preprocess"""
        response = requests.get(api_url)
        data = response.json()
        df = pd.DataFrame(data if isinstance(data, list) else data.get('results', []))
        
        numeric_columns = ['predicted_readmission_rate', 'expected_readmission_rate', 'excess_readmission_ratio']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        features = pd.concat([
            df[numeric_columns],
            pd.get_dummies(df['measure_name'], prefix='measure'),
            pd.DataFrame({'readmission_risk_ratio': df['predicted_readmission_rate'] / df['expected_readmission_rate']})
        ], axis=1)
        
        target = (df['excess_readmission_ratio'] > 1).astype(int)
        return features, target

    def train_models(self, X, y):
        """Train multiple models with different imputation strategies"""
        # Check if training_parameters.json exists
        if os.path.exists('training_parameters.json'):
            print("Found existing training_parameters.json. Skipping model training.")
            try:
                with open('training_parameters.json', 'r') as f:
                    training_params = json.load(f)
                print(f"Loaded parameters: {json.dumps(training_params, indent=2)}")
                return None, None  # Return None values since no new training was performed
            except Exception as e:
                print(f"Error loading training_parameters.json: {str(e)}")
                print("Proceeding with new model training...")
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        results = {}
        imputation_strategies = ['mean', 'median', 'most_frequent', 'constant']
        
        for strategy in imputation_strategies:
            print(f"\nTraining models with {strategy} imputation...")
            self.preprocessor = self.create_preprocessor(strategy)
            X_processed = {set_name: self.preprocessor.fit_transform(X_set) for set_name, X_set in zip(['train', 'val', 'test'], [X_train, X_val, X_test])}
            
            results[strategy] = self._train_strategy_models(X_processed, {'train': y_train, 'val': y_val, 'test': y_test})
        
        return self._create_summary(results)

    def _train_strategy_models(self, X_processed, y_sets):
        """Train models for a specific imputation strategy"""
        models_config = self.get_model_configs()
        strategy_results = {}
        
        for model_name, config in models_config.items():
            print(f"  Training {model_name}...")
            print(f"    Model parameters to search: {config['params']}")
            
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            try:
                grid_search.fit(X_processed['train'], y_sets['train'])
                print(f"    Best parameters found: {grid_search.best_params_}")
                print(f"    Best score: {grid_search.best_score_:.3f}")
                
                best_model = grid_search.best_estimator_
                self.models[model_name] = best_model
                
                strategy_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    **self._evaluate_sets(best_model, X_processed, y_sets)
                }
            except Exception as e:
                print(f"    Error training {model_name}: {str(e)}")
                continue
            
        return strategy_results

    def _evaluate_sets(self, model, X_processed, y_sets):
        """Evaluate model on all data sets"""
        results = {}
        for set_name in ['train', 'val', 'test']:
            pred = model.predict(X_processed[set_name])
            pred_proba = model.predict_proba(X_processed[set_name])[:, 1]
            results[f'{set_name}_scores'] = self.calculate_metrics(y_sets[set_name], pred, pred_proba)
        return results

    def _create_summary(self, results):
        """Create a summary of model performance and identify best model"""
        try:
            summary_rows = []
            best_model_metrics = {
                'val_roc_auc': 0,
                'imputation': None,
                'model': None,
                'metrics': None
            }
            
            print("\nProcessing model results...")
            print(f"Number of imputation strategies: {len(results)}")
            
            for strategy, models in results.items():
                print(f"Processing strategy: {strategy}, Number of models: {len(models)}")
                for model_name, metrics in models.items():
                    # Create summary row with the model name from the loop
                    summary_row = {
                        'Model': model_name,  # Use model_name from the loop instead of metrics['model']
                        'Imputation': strategy,
                        'Parameters': metrics['best_params']
                    }
                    
                    # Add scores for each phase
                    for phase in ['Train', 'Val', 'Test']:
                        phase_lower = phase.lower()
                        scores = metrics[f'{phase_lower}_scores']
                        for metric, value in scores.items():
                            summary_row[f'{phase} {metric}'] = value
                    
                    summary_rows.append(summary_row)
                    
                    # Track best model
                    if metrics['val_scores']['roc_auc'] > best_model_metrics['val_roc_auc']:
                        best_model_metrics['val_roc_auc'] = metrics['val_scores']['roc_auc']
                        best_model_metrics['imputation'] = strategy
                        best_model_metrics['model'] = model_name  # Use model_name from the loop
                        best_model_metrics['metrics'] = metrics
            
            # Verify we found a best model
            if best_model_metrics['metrics'] is None:
                raise ValueError("No best model was identified during training")
            
            # Debug print best model info
            print("\nBest model identified:")
            print(f"Model: {best_model_metrics['model']}")
            print(f"Imputation: {best_model_metrics['imputation']}")
            print(f"Parameters: {best_model_metrics['metrics']['best_params']}")
            
            # Save best model parameters
            try:
                best_params = best_model_metrics['metrics']['best_params']
                if not isinstance(best_params, dict):
                    raise ValueError(f"Invalid best parameters format: {best_params}")
                    
                with open('best_model_params.json', 'w') as f:
                    json.dump({
                        'model_type': best_model_metrics['model'],
                        'imputation_strategy': best_model_metrics['imputation'],
                        'parameters': best_params
                    }, f, indent=2)
                print("\nSuccessfully saved best_model_params.json")
                
            except Exception as e:
                print(f"\nError saving best_model_params.json: {str(e)}")
                print(f"Best model metrics: {best_model_metrics}")
                raise
            
            self.best_model_key = f"{best_model_metrics['model']}_{best_model_metrics['imputation']}"
            
            return results, pd.DataFrame(summary_rows)
            
        except Exception as e:
            print(f"\nError in _create_summary: {str(e)}")
            print("Current state of best_model_metrics:")
            print(best_model_metrics)
            raise

    def _plot_model_comparison(self, summary_df):
        """Create and save model comparison visualizations"""
        # Create output directory if it doesn't exist
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot ROC-AUC comparison
        fig = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=('ROC-AUC Scores by Model and Imputation Strategy',
                           'F1 Scores by Model and Imputation Strategy'),
            row_heights=[0.5, 0.5]
        )
        
        # Add ROC-AUC traces
        for i, phase in enumerate(['Train', 'Val', 'Test']):
            fig.add_trace(
                go.Bar(
                    name=f'{phase} ROC-AUC',
                    x=[f"{row['Model']} ({row['Imputation']})" for _, row in summary_df.iterrows()],
                    y=summary_df[f'{phase} ROC-AUC'],
                    text=summary_df[f'{phase} ROC-AUC'].round(3),
                    textposition='auto',
                ),
                row=1, col=1
            )
        
        # Add F1 traces
        for i, phase in enumerate(['Train', 'Val', 'Test']):
            fig.add_trace(
                go.Bar(
                    name=f'{phase} F1',
                    x=[f"{row['Model']} ({row['Imputation']})" for _, row in summary_df.iterrows()],
                    y=summary_df[f'{phase} F1'],
                    text=summary_df[f'{phase} F1'].round(3),
                    textposition='auto',
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Model Performance Comparison",
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update x-axis
        fig.update_xaxes(tickangle=45)
        
        # Save plots
        fig.write_html(f"{output_dir}/model_comparison.html")
        fig.write_image(f"{output_dir}/model_comparison.png", width=1200, height=1000)
        
        # Create detailed metrics heatmap
        metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
        phases = ['Train', 'Val', 'Test']
        
        # Prepare data for heatmap
        heatmap_data = []
        for metric in metrics:
            for phase in phases:
                column = f'{phase} {metric}'
                if column in summary_df.columns:
                    heatmap_data.append(summary_df[column])
        
        heatmap_df = pd.DataFrame(
            np.array(heatmap_data).T,
            columns=[f'{phase} {metric}' for metric in metrics for phase in phases],
            index=[f"{row['Model']} ({row['Imputation']})" for _, row in summary_df.iterrows()]
        )
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            text=np.round(heatmap_df.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='RdYlBu',
            hoverongaps=False
        ))
        
        # Update layout
        fig_heatmap.update_layout(
            title='Detailed Model Performance Metrics',
            height=600,
            width=1200,
            xaxis={'tickangle': 45}
        )
        
        # Save heatmap
        fig_heatmap.write_html(f"{output_dir}/model_comparison_heatmap.html")
        fig_heatmap.write_image(f"{output_dir}/model_comparison_heatmap.png", width=1200, height=600)

    def get_model_configs(self):
        try:
            # Try to load existing training configurations
            with open('model_params.json', 'r') as f:
                config = json.load(f)
                print("Found existing model parameters in model_params.json")
                
            # Convert configuration to model instances
            models_config = {}
            for model_name, model_config in config.items():
                model_type = model_config.get('type')
                model_class = self.model_classes.get(model_type)
                
                if model_class is None:
                    print(f"Warning: Unknown model type '{model_type}' for {model_name}. Using default LogisticRegression.")
                    model_class = LogisticRegression
                
                models_config[model_name] = {
                    'model': model_class(random_state=42),
                    'params': model_config.get('params', {})
                }
            
            return models_config
            
        except FileNotFoundError:
            print("model_params.json not found, running full pipeline with default parameters...")
            return self._get_default_config()

    def _get_model_type(self, model_name):
        """Helper method to get the proper model class name"""
        model_type_mapping = {
            'logistic': 'LogisticRegression',
            'random_forest': 'RandomForestClassifier',
            'xgboost': 'XGBClassifier'
        }
        return model_type_mapping.get(model_name, 'LogisticRegression')

    def _get_default_config(self):
        """Default model configuration when no parameters file exists"""
        return {
            'logistic': {
                'model': LogisticRegression(random_state=42),
                'params': {'C': [0.1, 1.0, 10.0]}
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7]
                }
            }
        }

class HospitalReadmissionEDA:
    def __init__(self, api_url):
        self.api_url = api_url
        self.df = None
        self.output_dir = 'plots'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """Load and preprocess data"""
        response = requests.get(self.api_url)
        data = response.json()
        self.df = pd.DataFrame(data if isinstance(data, list) else data.get('results', []))
        
        numeric_columns = ['predicted_readmission_rate', 'expected_readmission_rate', 'excess_readmission_ratio']
        self.df[numeric_columns] = self.df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        return self.df
    
    def generate_profile_report(self, output_file='hospital_readmission_profile.html'):
        """Generate comprehensive data profiling report"""
        profile = ProfileReport(self.df, title="Hospital Readmission Data Profiling Report", explorative=True)
        profile.to_file(output_file)
        
    def plot_readmission_distributions(self):
        """Plot distributions of readmission metrics"""
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Predicted Readmission Rate Distribution',
                                            'Expected Readmission Rate Distribution',
                                            'Excess Readmission Ratio Distribution',
                                            'Readmission Metrics Box Plot'))
        
        fig.add_trace(go.Histogram(x=self.df['predicted_readmission_rate'], name='Predicted'), row=1, col=1)
        fig.add_trace(go.Histogram(x=self.df['expected_readmission_rate'], name='Expected'), row=1, col=2)
        fig.add_trace(go.Histogram(x=self.df['excess_readmission_ratio'], name='Excess Ratio'), row=2, col=1)
        
        fig.add_trace(go.Box(y=self.df['predicted_readmission_rate'], name='Predicted'), row=2, col=2)
        fig.add_trace(go.Box(y=self.df['expected_readmission_rate'], name='Expected'), row=2, col=2)
        fig.add_trace(go.Box(y=self.df['excess_readmission_ratio'], name='Excess Ratio'), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Readmission Metrics Analysis")
        fig.write_html(f"{self.output_dir}/readmission_distributions.html")
        fig.write_image(f"{self.output_dir}/readmission_distributions.png")
        fig.show()
        
    def plot_measure_analysis(self):
        """Analyze readmission patterns by measure type"""
        measure_stats = self.df.groupby('measure_name').agg({
            'predicted_readmission_rate': 'mean',
            'expected_readmission_rate': 'mean',
            'excess_readmission_ratio': 'mean'
        }).round(3)
        
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=('Average Predicted Readmission Rate by Measure',
                                            'Average Expected Readmission Rate by Measure',
                                            'Average Excess Readmission Ratio by Measure'),
                            vertical_spacing=0.1)
        
        fig.add_trace(go.Bar(x=measure_stats.index, y=measure_stats['predicted_readmission_rate'], name='Predicted'), row=1, col=1)
        fig.add_trace(go.Bar(x=measure_stats.index, y=measure_stats['expected_readmission_rate'], name='Expected'), row=2, col=1)
        fig.add_trace(go.Bar(x=measure_stats.index, y=measure_stats['excess_readmission_ratio'], name='Excess Ratio'), row=3, col=1)
        
        fig.update_layout(height=1000, showlegend=False, title_text="Readmission Metrics by Measure Type")
        fig.write_html(f"{self.output_dir}/measure_analysis.html")
        fig.write_image(f"{self.output_dir}/measure_analysis.png")
        fig.show()
        
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for numeric variables"""
        numeric_cols = ['predicted_readmission_rate', 'expected_readmission_rate', 'excess_readmission_ratio']
        corr_matrix = self.df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdBu'
        ))
        
        fig.update_layout(title='Correlation Heatmap of Readmission Metrics', height=500, width=700)
        fig.write_html(f"{self.output_dir}/correlation_heatmap.html")
        fig.write_image(f"{self.output_dir}/correlation_heatmap.png")
        fig.show()
        
    def plot_scatter_matrix(self):
        """Create scatter plot matrix for numeric variables"""
        numeric_cols = ['predicted_readmission_rate', 'expected_readmission_rate', 'excess_readmission_ratio']
        fig = px.scatter_matrix(self.df, dimensions=numeric_cols, title="Scatter Plot Matrix of Readmission Metrics")
        fig.update_layout(height=800)
        fig.write_html(f"{self.output_dir}/scatter_matrix.html")
        fig.write_image(f"{self.output_dir}/scatter_matrix.png")
        fig.show()
        
    def plot_time_series(self):
        """Plot time series analysis if temporal data is available"""
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            fig = make_subplots(rows=3, cols=1,
                                subplot_titles=('Predicted Readmission Rate Over Time',
                                                'Expected Readmission Rate Over Time',
                                                'Excess Readmission Ratio Over Time'))
            
            metrics = ['predicted_readmission_rate', 'expected_readmission_rate', 'excess_readmission_ratio']
            
            for i, metric in enumerate(metrics, 1):
                daily_avg = self.df.groupby('date')[metric].mean().reset_index()
                fig.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg[metric], mode='lines+markers', name=metric), row=i, col=1)
            
            fig.update_layout(height=900, showlegend=False, title_text="Readmission Metrics Over Time")
            fig.write_html(f"{self.output_dir}/time_series.html")
            fig.write_image(f"{self.output_dir}/time_series.png")
            fig.show()
            
    def run_full_eda(self):
        """Run complete EDA pipeline"""
        print("Loading data...")
        self.load_data()
        
        print("\nGenerating comprehensive data profile report...")
        self.generate_profile_report()
        
        print("\nPlotting readmission distributions...")
        self.plot_readmission_distributions()
        
        print("\nAnalyzing measures...")
        self.plot_measure_analysis()
        
        print("\nGenerating correlation analysis...")
        self.plot_correlation_heatmap()
        
        print("\nCreating scatter plot matrix...")
        self.plot_scatter_matrix()
        
        if 'date' in self.df.columns:
            print("\nPerforming time series analysis...")
            self.plot_time_series()

def main():
    api_url = "https://data.cms.gov/provider-data/api/1/datastore/query/9n3s-kdb3/0"
    eda = HospitalReadmissionEDA(api_url)
    eda.run_full_eda()

    predictor = HospitalReadmissionPredictor()
    features, target = predictor.load_and_preprocess_data(api_url)
    
    try:
        # Train models (will use parameters from model_params.json if it exists)
        results, summary_df = predictor.train_models(features, target)
        
        # Save the best model
        if hasattr(predictor, 'best_model_key'):
            # Extract the full model name (e.g., 'random_forest' instead of just 'random')
            model_name = predictor.best_model_key.split('_')[0]
            if model_name == 'random':  # Fix for random_forest
                model_name = 'random_forest'
            
            imputation_strategy = predictor.best_model_key.split('_', 1)[1] if '_' in predictor.best_model_key else None
            
            if model_name in predictor.models:
                best_model = predictor.models[model_name]
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(best_model, f)
                print(f"Saved best model '{model_name}' to best_model.pkl")
                
                # Save training parameters
                with open('training_parameters.json', 'w') as f:
                    json.dump({
                        'imputation_strategy': imputation_strategy,
                        'model_name': model_name,
                        'model_parameters': best_model.get_params()
                    }, f, indent=2)
                print("Saved training parameters to training_parameters.json")
            else:
                print(f"Warning: Best model '{model_name}' not found in trained models")
                print(f"Available models: {list(predictor.models.keys())}")
            
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
