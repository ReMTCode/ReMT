import pandas as pd
from joblib import load
from data_processing import process_data
from consistency import check_consistency

data_path = 'data/data.xlsx'
output_dir = 'data/'
model_path = 'model/gbdt_model.joblib'
consistency_results_path = output_dir + 'consistency_results.csv'

process_data(data_path, output_dir)

translated_data_paths = [output_dir + 'esnewdata.csv', output_dir + 'itnewdata.csv', output_dir + 'renewdata.csv', output_dir + 'GAnewdata.csv']
check_consistency(output_dir + 'sourcenewdata.csv', translated_data_paths, consistency_results_path)

print("Loading the GBDT model...")
gbdt = load(model_path)

new_data_pd = pd.read_csv(consistency_results_path)

new_features = new_data_pd.iloc[:, 0:10].values

new_predictions = gbdt.predict(new_features)

new_data_pd['Predictions'] = new_predictions
new_data_pd.to_csv(output_dir + 'new_predictions.csv', index=False)