import csv
import requests
import json
import subprocess
import sys
import os

OLLAMA_MODEL = 'llama3'

def generate_prompt_short(premise, hypothesis):
    prompt = f"""
The logical relationship between the a premise and a hypothesis is defined as one of the following:
Entailment means the hypothesis is definitely true given the premise.
Contradiction means the hypothesis is definitely false given the premise.
Neutral means the hypothesis might be a true description of the premise, but there is no direct evidence to support it.

What is the logical relationship between the following premise and hypothesis?
Your answer should strictly follow the standard parseable JSON format: {{"reason": "<your_reason>", "relationship": "<your_answer>"}}, where your reason is a detailed step by step chain of thought through the question and your_answer should strictly be one word - entailment, neutral, or contradiction.
Avoid any additional text outside of this format.

Premise: {premise} 
Hypothesis: {hypothesis}

Answer:
"""
    return prompt

def generate_prompt_long(premise, hypothesis):
    prompt = f"""
The logical relationship between the following premise and hypothesis is defined as one of the following:

contradiction: the hypothesis is definitely false given the premise, you can reject the hypothesis based on the premise;
entailment: the hypothesis is definitely true given the premise, you can accept the hypothesis based on the premise.
neutral: the hypothesis might be a true description of the premise, but there is no direct evidence to support it, you can neither accept nor reject the hypothesis based on the premise;

Examples:
    Question:
    Premise: A boy is drinking out of a water fountain shaped like a woman.
    Hypothesis: A man is drinking lemonade from a glass.

    Answer:
    {{"cot": "1. Premise Analysis: The premise states that 'A boy is drinking out of a water fountain shaped like a woman.' This provides specific information that a boy is drinking from a water fountain, not from a glass or any other container, and it mentions the type of drink is water. 2. Hypothesis Analysis: The hypothesis says 'A man is drinking lemonade from a glass.' This introduces new information: the drink is lemonade from a glass. 3. Comparing the Two: The premise clearly states the that the boy is drinking from a water fountain, not a glass, and it confirms the drink is water. 4. Conclusion: The hypothesis directly contradicts the premise because the drink, and the container are both different. Therefore, the relationship is contradiction.", "relationship": "contradiction"}}

    Question:
    Premise: A boy is drinking out of a water fountain shaped like a woman.
    Hypothesis: A male is getting a drink of water.

    Answer:
    {{"cot": "1. Premise Analysis: The premise states that 'A boy is drinking out of a water fountain shaped like a woman.' This clearly describes a male (the boy) who is drinking water. 2. Hypothesis Analysis: The hypothesis says 'A male is getting a drink of water.' This aligns directly with the premise, as the boy is male and is drinking water. 3. Comparing the Two: The premise directly supports the hypothesis, as the action and the subject (a male drinking water) are explicitly mentioned. 4. Conclusion: The hypothesis is definitely true given the premise, so the relationship is entailment.", "relationship": "entailment"}}

    Question:
    Premise: A boy is drinking out of a water fountain shaped like a woman.
    Hypothesis: A sculptor takes a drink from a fountain that he made that looks like his girlfriend.

    Answer:
    {{"cot": "1. Premise Analysis: The premise states 'A boy is drinking out of a water fountain shaped like a woman.' It describes a boy drinking from a fountain, with no information about who made the fountain or the boy's relationship to it. 2. Hypothesis Analysis: The hypothesis says 'A sculptor takes a drink from a fountain that he made that looks like his girlfriend.' It introduces two additional details: the person drinking is a sculptor, and the fountain was made by him to resemble his girlfriend. 3. Comparing the Two: The premise provides no direct information about the boy being a sculptor, nor that he created the fountain or that it looks like his girlfriend. While this scenario might be possible, the premise does not offer any evidence for the hypothesis. 4. Conclusion: Since the hypothesis might be true but there is no direct evidence to support it, the relationship is neutral.", "relationship": "neutral"}}

What is the logical relationship between the following premise and hypothesis? Your answer should: 1. strictly follow the standard parseable JSON format: {{"cot": <your chain-of-though>, "relationship": <your answer>}}; 2. contain only the answer part, avoid using stuff like 'Let's analyze the premise and hypothesis:' or 'Here is my answer:'.

Question:
Premise: {premise} 
Hypothesis: {hypothesis}

Answer:
"""
    return prompt

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

def query_ollama_cmd(prompt, model=OLLAMA_MODEL):
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
    
    message = result.stdout.strip()
    print(message)
    return message

def parse_response(response_text):
    try:
        response_text_processed = json.loads(response_text)
        response_text_processed = response_text_processed['relationship'].lower()
    except Exception as e:
        print(f'Failed to load response text as json and get relationship. Message: {e}')
        return -1
            
    if 'entailment' == response_text_processed:
        response = 0
    elif 'neutral' == response_text_processed:
        response = 1
    elif 'contradiction' == response_text_processed:
        response = 2
    else:
        response = -1
    
    return response


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

                prompt = generate_prompt_short(premise, hypothesis)
                response_text = query_ollama_cmd(prompt, model=model)
                response = parse_response(response_text)
                if response == -1:                    
                    print(f'Failed to parse response at index {index}. Skipping.')
                    print(f'Response is {response_text}')
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