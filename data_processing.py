import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
from googletrans import Translator
import openai

def generate_data(model, tokenizer, question, prompt):
    text = prompt + question + " Answer:"
    model_inputs = tokenizer([text], return_tensors="pt")
    outputs = model.generate(**model_inputs, min_new_tokens=1, max_new_tokens=100, do_sample=True, temperature=0.7, output_scores=True, return_dict_in_generate=True, eos_token_id=2)
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    input_length = model_inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    answer = tokenizer.batch_decode(generated_tokens)[0].replace('</s>', '')

    probs = []
    tokens = []
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        gen = tokenizer.decode(tok)
        if gen in ["\n", "</s>"]:
            continue
        if gen in ["(", "#", "|"]:
            break
        tokens.append(gen)
        probs.append(np.exp(score.numpy()))
    if tokens and tokens[-1] == ".":
        tokens.pop()
        probs.pop()
    return answer, probs, tokens

def calculate_scores(probs):
    if not probs:
        return [-1, -1]
    return [np.mean(probs), np.prod(probs) ** (1/len(probs))]

def process_data(data_path, output_dir):
    openai.api_key = 'your_openai_api_key'

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    access_token = "" 
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config, token=access_token)

    print(f"Successfully loaded the model {model_name} into memory")

    data_pd = pd.read_excel(data_path)
    questions = data_pd.iloc[:, 1].tolist()

    prompt_en = "You're required to answer a question. USE ONE WORD OR ONE PHRASE TO ANSWER. IF you don't know the answer, simply write '(IDN)' without any additional text or explanation. Don't explain. Don't generate new question. Don't summarize. Don't give notes or annotations. Question:"
    results_en = []
    for i, question in enumerate(questions):
        answer, probs, _ = generate_data(model, tokenizer, question, prompt_en)
        scores = calculate_scores(probs)
        results_en.append([i+1, question, answer, scores[0], scores[1]])
    headers_en = ["Id", "Question", "Output", "AvgScore", "NormScore"]
    with open(f"{output_dir}sourcenewdata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers_en)
        writer.writerows(results_en)

    translator = Translator()
    translated_questions_es = [translator.translate(question, dest='es').text for question in questions]
    prompt_es = "Tienes que responder a una pregunta en inglés. UTILICE UNA PALABRA O UNA FRASE PARA RESPONDER. Si no conoce la respuesta, escriba simplemente «(IDN)» sin ningún texto o explicación adicional. No dé explicaciones. No genere una nueva pregunta. No resumas. No ponga notas ni anotaciones. Pregunta:"
    results_es = []
    for i, question in enumerate(translated_questions_es):
        answer, probs, _ = generate_data(model, tokenizer, question, prompt_es)
        scores = calculate_scores(probs)
        results_es.append([i+1, question, answer, scores[0], scores[1]])
    headers_es = ["Id", "Translated Question", "Answer", "AvgScore", "NormScore"]
    with open(f"{output_dir}esnewdata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers_es)
        writer.writerows(results_es)

    translated_questions_it = [translator.translate(question, dest='it').text for question in questions]
    prompt_it = "Dovete rispondere a una domanda in inglese. PER RISPONDERE USATE UNA PAROLA O UNA FRASE. Se non conoscete la risposta, scrivete semplicemente ‘(IDN)’ senza alcun testo aggiuntivo o spiegazione. Non dare spiegazioni. Non generare nuove domande. Non riassumere. Non fornire note o annotazioni. Domanda:"
    results_it = []
    for i, question in enumerate(translated_questions_it):
        answer, probs, _ = generate_data(model, tokenizer, question, prompt_it)
        scores = calculate_scores(probs)
        results_it.append([i+1, question, answer, scores[0], scores[1]])
    headers_it = ["Id", "Translated Question", "Answer", "AvgScore", "NormScore"]
    with open(f"{output_dir}itnewdata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers_it)
        writer.writerows(results_it)

    prompt_template = (
        "I hope you can serve as a copywriter and text polishing officer. "
        "I will send you the English text, and you can help me improve the version. "
        "I hope you can change the words and structure of the sentence as much as possible. "
        "Keep the same meaning, but make them easier to understand. "
        "You only need to polish the content without explaining the questions and requirements raised in the content."
    )

    def improve_text(question):
        prompt = prompt_template + "\n\n" + question
        response = openai.Completion.create(
            engine="text-davinci-002",  # Use the gpt-3.5 model
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n\n"]
        )
        return response.choices[0].text.strip()

    results_re = []
    for i, question in enumerate(questions):
        improved_question = improve_text(question)
        answer, probs, _ = generate_data(model, tokenizer, improved_question, prompt_en)
        scores = calculate_scores(probs)
        results_re.append([i+1, question, improved_question, answer, scores[0], scores[1]])
    headers_re = ["Id", "Original Question", "Improved Question", "Answer", "AvgScore", "NormScore"]
    with open(f"{output_dir}renewdata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers_re)
        writer.writerows(results_re)

    prompt_ga = (
        "| CONTEXT |\n"
        "You're a knowledge expert and you're trying to solve a real-world question that I've posed.\n"
        "| AUDIENCE |\n"
        "The audience to whom you are answering the question is the reviewer, and the consequences of answering inappropriately "
        "or spreading false rumors will be severe. Please tailor your answer to the audience. Simply replying '(IDN)' to uncertain questions will avoid serious consequences.\n"
        "| RESPONSE |\n"
        "A word or a phrase.\n"
        "| OBJECTIVE |\n"
        "Correctly answer the question I've given below.\n"
        "You will be penalized if you answer incorrectly. "
        "If you are not sure enough about the answer to the question, simply write '(IDN)' without any additional text "
        "or explanation so that you are not penalized.\n"
        "| QUESTION |"
    )

    results_ga = []
    for i, question in enumerate(questions):
        answer, probs, _ = generate_data(model, tokenizer, question, prompt_ga)
        scores = calculate_scores(probs)
        results_ga.append([i+1, question, answer, scores[0], scores[1]])
    headers_ga = ["Id", "Question", "Answer", "AvgScore", "NormScore"]
    with open(f"{output_dir}GAnewdata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers_ga)
        writer.writerows(results_ga)

if __name__ == "__main__":
    process_data('data/data.xlsx', 'data/')