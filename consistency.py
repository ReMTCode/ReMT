import pandas as pd
import openai
import csv

def contains_uncertainty(text):
    prompt = "Does the following statement contain the meaning 'I am not sure'or 'I don't know'? If it does, output 0, if it does not, output 1. ('IDN' means I don't know in shorthand.)(Note: the output can only be 0 or 1 without any interpretation).(Note: If an answer is divided into multiple answers using separators, only the first answer will be considered.)"
    prompt += "\n\n" + text
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n\n"]
    )
    return int(response.choices[0].text.strip())

def are_answers_consistent(question, answer1, answer2):
    prompt = "Are the two answers to the question consistent? Do not judge the rightness or wrongness of the answer itself. If it is, output 0; if not, output 1. (Note: Yes and Right have the same meaning; No and Wrong have the same meaning.) (Note: the output can only be 0 or 1, no explanation is given).(Note: If an answer is divided into multiple answers using separators, only the first answer will be considered.)"
    prompt += "\n\nQuestion: " + question + "\nAnswer 1: " + answer1 + "\nAnswer 2: " + answer2
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n\n"]
    )
    return int(response.choices[0].text.strip())

def check_consistency(source_data_path, translated_data_paths, output_path):
    openai.api_key = 'your_openai_api_key'

    # Read the source dataset and take the absolute value of AvgScore and NormScore
    df_en = pd.read_csv(source_data_path)
    df_en['AvgScore'] = df_en['AvgScore'].abs()
    df_en['NormScore'] = df_en['NormScore'].abs()

    # Read the translated datasets and take the absolute value of AvgScore and NormScore
    df_es = pd.read_csv(translated_data_paths[0])
    df_es['AvgScore'] = df_es['AvgScore'].abs()
    df_es['NormScore'] = df_es['NormScore'].abs()
    
    df_it = pd.read_csv(translated_data_paths[1])
    df_it['AvgScore'] = df_it['AvgScore'].abs()
    df_it['NormScore'] = df_it['NormScore'].abs()

    df_re = pd.read_csv(translated_data_paths[2])
    df_re['AvgScore'] = df_re['AvgScore'].abs()
    df_re['NormScore'] = df_re['NormScore'].abs()

    df_ga = pd.read_csv(translated_data_paths[3])
    df_ga['AvgScore'] = df_ga['AvgScore'].abs()
    df_ga['NormScore'] = df_ga['NormScore'].abs()

    # Initialize the results list
    results = []

    # Iterate over the source dataset
    for index, row_en in df_en.iterrows():
        uncertainty_score = contains_uncertainty(row_en['Output'])
        if uncertainty_score == 0:
            diff_avg_score_es = 0
            diff_norm_score_es = 0
            diff_avg_score_it = 0
            diff_norm_score_it = 0
            diff_avg_score_re = 0
            diff_norm_score_re = 0
            diff_avg_score_ga = 0
            diff_norm_score_ga = 0
        else:
            corresponding_row_es = df_es[df_es['Id'] == row_en['Id']]
            corresponding_row_it = df_it[df_it['Id'] == row_en['Id']]
            corresponding_row_re = df_re[df_re['Id'] == row_en['Id']]
            corresponding_row_ga = df_ga[df_ga['Id'] == row_en['Id']]
            
            # ES
            if not corresponding_row_es.empty:
                consistency_score_es = are_answers_consistent(row_en['Question'], row_en['Output'], corresponding_row_es['Answer'].values[0])
                if consistency_score_es == 0:
                    diff_avg_score_es = row_en['AvgScore']
                    diff_norm_score_es = row_en['NormScore']
                else:
                    diff_avg_score_es = abs(row_en['AvgScore'] - corresponding_row_es['AvgScore'].values[0])
                    diff_norm_score_es = abs(row_en['NormScore'] - corresponding_row_es['NormScore'].values[0])
            else:
                diff_avg_score_es = 0
                diff_norm_score_es = 0

            # IT
            if not corresponding_row_it.empty:
                consistency_score_it = are_answers_consistent(row_en['Question'], row_en['Output'], corresponding_row_it['Answer'].values[0])
                if consistency_score_it == 0:
                    diff_avg_score_it = row_en['AvgScore']
                    diff_norm_score_it = row_en['NormScore']
                else:
                    diff_avg_score_it = abs(row_en['AvgScore'] - corresponding_row_it['AvgScore'].values[0])
                    diff_norm_score_it = abs(row_en['NormScore'] - corresponding_row_it['NormScore'].values[0])
            else:
                diff_avg_score_it = 0
                diff_norm_score_it = 0

            # RE
            if not corresponding_row_re.empty:
                consistency_score_re = are_answers_consistent(row_en['Question'], row_en['Output'], corresponding_row_re['Answer'].values[0])
                if consistency_score_re == 0:
                    diff_avg_score_re = row_en['AvgScore']
                    diff_norm_score_re = row_en['NormScore']
                else:
                    diff_avg_score_re = abs(row_en['AvgScore'] - corresponding_row_re['AvgScore'].values[0])
                    diff_norm_score_re = abs(row_en['NormScore'] - corresponding_row_re['NormScore'].values[0])
            else:
                diff_avg_score_re = 0
                diff_norm_score_re = 0

            # GA
            if not corresponding_row_ga.empty:
                consistency_score_ga = are_answers_consistent(row_en['Question'], row_en['Output'], corresponding_row_ga['Answer'].values[0])
                if consistency_score_ga == 0:
                    diff_avg_score_ga = row_en['AvgScore']
                    diff_norm_score_ga = row_en['NormScore']
                else:
                    diff_avg_score_ga = abs(row_en['AvgScore'] - corresponding_row_ga['AvgScore'].values[0])
                    diff_norm_score_ga = abs(row_en['NormScore'] - corresponding_row_ga['NormScore'].values[0])
            else:
                diff_avg_score_ga = 0
                diff_norm_score_ga = 0

        # Append the results to the results list
        results.append([row_en['AvgScore'], row_en['NormScore'], diff_avg_score_es, diff_norm_score_es, diff_avg_score_it, diff_norm_score_it, diff_avg_score_re, diff_norm_score_re, diff_avg_score_ga, diff_norm_score_ga])

    # Write the results to a new CSV file
    headers = ["AvgScore_en", "NormScore_en", "es_Diff_AvgScore", "es_Diff_NormScore", "it_Diff_AvgScore", "it_Diff_NormScore", "re_Diff_AvgScore", "re_Diff_NormScore", "ga_Diff_AvgScore", "ga_Diff_NormScore"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

# If this script is run directly, check consistency
if __name__ == "__main__":
    check_consistency('data/sourcenewdata.csv', ['data/esnewdata.csv', 'data/itnewdata.csv', 'data/renewdata.csv', 'data/GAnewdata.csv'], 'data/consistency_results.csv')