import csv
import requests
import json
import subprocess
import sys
import os

OLLAMA_MODEL = 'llama3.2'

def query_ollama_api(premise, hypothesis, model=OLLAMA_MODEL, server_url='http://localhost:11434/api/generate'):
    prompt = (
        f"What is the logical relationship between the following premise and hypothesis "
        f"(one of entailment, neutral, or contradiction)?\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"Reply only one word.\n"
        f"Answer:"
    )

    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        generated_text = ''
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                if 'response' in data:
                    generated_text += data['response']

        return generated_text.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama API: {e}")
        return "Error querying model."

def query_ollama_cmd(premise, hypothesis, model=OLLAMA_MODEL):
    prompt = f"""What is the logical relationship between the following premise and hypothesis (one of entailment, neutral, or contradiction)? Your answer should strictly follow the given example format and should contain only the answer part. 

Entailment means the hypothesis is definitely a true description of the premise;

Contradiction means the hypothesis is definitely false given the premise;

Neutral means the hypothesis is might be a true description of the premise, but there is no direct evidence to support it.

For example:

Question:
Premise: A man in a tank top fixing himself a hotdog.
Hypothesis: The child was happy.

Answer: {{"relationship": "neutral","reason": "There's no direct link suggesting that the man’s action would make the child happy, nor is there any contradiction between the two. They describe different subjects and unrelated situations. Hence, they are neutral."}}

You are asked to answer the following question:

Premise: {premise} 
Hypothesis: {hypothesis}

Answer:"""

    if os.name == 'nt':  # Windows
        command = f'ollama run {model}'
        result = subprocess.run(
            command,
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            shell=True
        )
    else:  # Unix/Linux/MacOS
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            text=True
        )

    if result.stderr:
        print(f"Error querying Ollama: {result.stderr}", file=sys.stderr)
    
    return result.stdout.strip()

def process_csv(input_path, output_path, model=OLLAMA_MODEL):
    correct_count = 0
    total_count = 0
    with open(input_path, 'r', newline='', encoding='utf-8') as input:
        reader = csv.DictReader(input)
        fieldnames = reader.fieldnames + ['response']

        with open(output_path, 'w', newline='', encoding='utf-8') as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                premise = row['premise']
                hypothesis = row['hypothesis']
                index = row['index']
                true_label = row['true_label']

                response_text = query_ollama_cmd(premise, hypothesis, model=model).lower()
                try:
                    response_relationship = json.loads(response_text)
                    response_relationship = response_relationship['relationship']
                except Exception as e:
                    print(f'Failed to load response text as json and get relationship. Message: {e}')
                    continue
                if 'entailment' in response_relationship:
                    response = 0
                elif 'neutral' in response_relationship:
                    response = 1
                elif 'contradiction' in response_relationship:
                    response = 2
                else:
                    print(f'Malformed response: index {index}, skipping.')
                    continue

                row['response'] = response

                total_count += 1
                if int(true_label) == response:
                    correct_count += 1

                writer.writerow(row)
                print(f'Processed index: {index}, response is {response}')

    return total_count, correct_count

if __name__ == '__main__':
    input_path = 'intersection_output_ignore_prediction.csv'
    output_path = f'{OLLAMA_MODEL}_responses.csv'
    total, correct = process_csv(input_path, output_path)
    print(f'Finished querying {OLLAMA_MODEL}, total examples {total}, correct predictions {correct}, accuracy {(correct/total):.4f}')