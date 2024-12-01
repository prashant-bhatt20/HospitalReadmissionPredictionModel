import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

class HospitalReadmissionEDA:
    def __init__(self, api_url):
        self.api_url = api_url
        self.df = None
        sns.set_style("whitegrid")
        
    def load_data(self):
        """Load and prepare data from CMS API"""
        print("Loading data from CMS API...")
        response = requests.get(self.api_url)
        data = response.json()
        
        # Convert to DataFrame
        if isinstance(data, list):
            self.df = pd.DataFrame(data)
        else:
            self.df = pd.DataFrame(data.get('results', []))
            
        # Convert numeric columns
        numeric_columns = ['predicted_readmission_rate', 'expected_readmission_rate', 
                         'excess_readmission_ratio']
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        print(f"Dataset loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        return self.df
    
    def basic_stats(self):
        """Generate basic statistical summary"""
        print("\n=== Basic Statistical Summary ===")
        
        # Dataset info
        print("\nDataset Info:")
        print("-" * 50)
        print(self.df.info())
        
        # Numerical summary
        print("\nNumerical Summary:")
        print("-" * 50)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())
        
        # Missing values
        print("\nMissing Values Summary:")
        print("-" * 50)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Values'] > 0])
        
    def distribution_analysis(self):
        """Analyze distributions of key metrics"""
        print("\n=== Distribution Analysis ===")
        
        metrics = ['predicted_readmission_rate', 'expected_readmission_rate', 
                  'excess_readmission_ratio']
        
        fig, axes = plt.subplots(len(metrics), 2, figsize=(15, 5*len(metrics)))
        
        for i, metric in enumerate(metrics):
            # Histogram
            sns.histplot(data=self.df, x=metric, ax=axes[i,0])
            axes[i,0].set_title(f'Distribution of {metric}')
            
            # QQ Plot
            stats.probplot(self.df[metric].dropna(), dist="norm", plot=axes[i,1])
            axes[i,1].set_title(f'Q-Q Plot of {metric}')
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests for normality
        print("\nNormality Tests (Shapiro-Wilk):")
        print("-" * 50)
        for metric in metrics:
            stat, p_value = stats.shapiro(self.df[metric].dropna())
            print(f"{metric}:")
            print(f"Statistic: {stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Normal distribution: {p_value > 0.05}\n")
    
    def correlation_analysis(self):
        """Analyze correlations between numerical variables"""
        print("\n=== Correlation Analysis ===")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Variables')
        plt.show()
        
        # Print strong correlations
        print("\nStrong Correlations (|r| > 0.5):")
        print("-" * 50)
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j]) > 0.5:
                    print(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_matrix.iloc[i,j]:.3f}")
    
    def measure_analysis(self):
        """Analyze readmission patterns across different measures"""
        print("\n=== Measure Analysis ===")
        
        # Count of each measure
        measure_counts = self.df['measure_name'].value_counts()
        
        # Plot measure distribution
        plt.figure(figsize=(12, 6))
        measure_counts.plot(kind='bar')
        plt.title('Distribution of Measures')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Statistics by measure
        print("\nReadmission Statistics by Measure:")
        print("-" * 50)
        measure_stats = self.df.groupby('measure_name').agg({
            'predicted_readmission_rate': ['mean', 'std', 'count'],
            'excess_readmission_ratio': ['mean', 'std']
        }).round(3)
        print(measure_stats)
        
        # ANOVA test for differences between measures
        measures = self.df['measure_name'].unique()
        f_stat, p_value = stats.f_oneway(*[
            self.df[self.df['measure_name'] == measure]['excess_readmission_ratio'].dropna()
            for measure in measures
        ])
        
        print("\nANOVA Test for Differences Between Measures:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Significant differences exist: {p_value < 0.05}")
    
    def readmission_risk_analysis(self):
        """Analyze readmission risk patterns"""
        print("\n=== Readmission Risk Analysis ===")
        
        # Create risk categories
        self.df['risk_category'] = pd.qcut(
            self.df['predicted_readmission_rate'],
            q=4,
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        # Plot risk distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='risk_category', y='excess_readmission_ratio')
        plt.title('Excess Readmission Ratio by Risk Category')
        plt.tight_layout()
        plt.show()
        
        # Risk category statistics
        print("\nStatistics by Risk Category:")
        print("-" * 50)
        risk_stats = self.df.groupby('risk_category').agg({
            'excess_readmission_ratio': ['mean', 'std', 'count']
        }).round(3)
        print(risk_stats)
        
        # Chi-square test for independence
        high_risk = self.df['risk_category'] == 'High'
        excess_readmission = self.df['excess_readmission_ratio'] > 1
        
        contingency = pd.crosstab(high_risk, excess_readmission)
        chi2, p_value = stats.chi2_contingency(contingency)[:2]
        
        print("\nChi-square Test for Risk Category vs Excess Readmission:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Significant association exists: {p_value < 0.05}")
    
    def run_full_eda(self):
        """Run all EDA analyses"""
        self.load_data()
        self.basic_stats()
        self.distribution_analysis()
        self.correlation_analysis()
        self.measure_analysis()
        self.readmission_risk_analysis()
        
        # Save figures
        plt.close('all')

def main():
    api_url = "https://data.cms.gov/provider-data/api/1/datastore/query/9n3s-kdb3/0"
    eda = HospitalReadmissionEDA(api_url)
    eda.run_full_eda()

if __name__ == "__main__":
    main()