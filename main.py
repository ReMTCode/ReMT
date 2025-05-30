import pandas as pd
from joblib import load
from data_processing import process_data
from consistency import check_consistency

# Define the paths for the data and model
data_path = 'data/data.xlsx'
output_dir = 'data/'
model_path = 'model/gbdt_model.joblib'
consistency_results_path = output_dir + 'consistency_results.csv'

# Process the data to generate various CSV files including sourcenewdata.csv, esnewdata.csv, itnewdata.csv, renewdata.csv, and GAnewdata.csv
process_data(data_path, output_dir)

# Check consistency across different datasets and generate consistency_results.csv
translated_data_paths = [output_dir + 'esnewdata.csv', output_dir + 'itnewdata.csv', output_dir + 'renewdata.csv', output_dir + 'GAnewdata.csv']
check_consistency(output_dir + 'sourcenewdata.csv', translated_data_paths, consistency_results_path)

# Load the GBDT model for predictions
print("Loading the GBDT model...")
gbdt = load(model_path)

# Read the new dataset where consistency has been checked
new_data_pd = pd.read_csv(consistency_results_path)

# Assume the first 10 columns of the new dataset are feature columns
new_features = new_data_pd.iloc[:, 0:10].values

# Use the GBDT model to make predictions based on the feature columns
new_predictions = gbdt.predict(new_features)

# Save the prediction results to a new CSV file
new_data_pd['Predictions'] = new_predictions
new_data_pd.to_csv(output_dir + 'new_predictions.csv', index=False)