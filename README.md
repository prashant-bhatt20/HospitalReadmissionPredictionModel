Hospital Readmission Prediction System 

What Does This Do? 
This system helps hospitals predict whether patients might need to come back to the hospital after they leave. It's like having a smart helper that learns from past patient information to make better guesses about future patients!

Setup & Installation 

1. Prerequisites
Make sure you have these installed on your computer:
- Python 3.8 or higher
- pip (Python package installer)

2. Clone the Repository
- bash
- git clone https://github.com/prashant-bhatt20/HospitalReadmissionPredictionModel.git
- cd hospital-readmission-predictor

3. Create a Virtual Environment (Recommended)
- python -m venv hospital-readmission-env
- source hospital-readmission-env/bin/activate

4. Install Dependencies
- pip install -r requirements.txt

5. Run the Program
- python app.py
 
6. Expected Output 

 1. Folders Created
 HospitalReadmissionPredictionModel/
│
├── plots/
│ ├── model_comparison.html
│ ├── model_comparison.png
│ ├── correlation_heatmap.html
│ ├── correlation_heatmap.png
│ └── ...
│
├── best_model.pkl
├── training_parameters.json
├── model_comparison.csv
└── hospital_readmission_profile.html

1. Getting and Understanding Data
- Fetches hospital data from a special healthcare website (CMS)
- Looks at three main things:
  - How many patients we think will come back
  - How many patients usually come back
  - How different these numbers are from each other

2. Creating EDA charts
The system creates lots of helpful charts to understand the data better:
- Shows how often patients come back to different hospitals
- Compares different types of hospital visits
- Makes colorful maps showing how different numbers are related
- Tracks changes over time (if we have dates)

3. Teaching Computers to Predict 
Uses four different types of smart computer programs (models):
- Simple calculator (Logistic Regression)
- Forest of decision trees (Random Forest)
- Super-powered learner (XGBoost)
- Gradient booster (like a turbo engine for learning)

4. Testing and Comparing 
- Splits data into three parts:
  - Training data (75%) - to teach the computer
  - Validation data (15%) - to check if it learned well
  - Testing data (10%) - for final testing

5. Saving the Best Results 
- Saves the best performing model
- Creates reports showing how well each model did
- Makes comparison charts to see which model was the best

What Files Does It Create? 📁
- 'plots/' folder with lots of helpful charts
- 'best_model.pkl' - the best prediction helper
- 'training_parameters.json' - settings that worked best
- 'model_comparison.csv' - spreadsheet comparing all models
- 'hospital_readmission_profile.html' - detailed report about the data

How to Use It? 
1. Make sure you have all the required Python packages
2. Run the program
3. Look at the charts and reports it creates
4. Use the best model to make predictions

WHats included
- Tries different ways to handle missing information
- Tests multiple types of prediction models
- Creates easy-to-understand visual reports
- Automatically saves the best performing model
- Uses real hospital data from CMS

