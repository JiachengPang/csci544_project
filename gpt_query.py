import openai_key as key
import ollama_query
import csv
from openai import OpenAI

GPT_MODEL = 'gpt-4o-mini'
client = OpenAI(api_key=key.OPENAI_KEY)

def query_gpt(query, client):
    print(f'Querying {GPT_MODEL} with the following promt.')
    print(query)

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )

    message = response.choices[0].message.content
    print('--------------------------------------------------------------')
    print('Response message is')
    print(message)
    print('--------------------------------------------------------------')
    return message


def process_csv_gpt(input_path, output_path, model=GPT_MODEL):
    print(f'Starting to process {model} on {input_path}')
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

                prompt = ollama_query.generate_prompt_short(premise, hypothesis)
                response_text = query_gpt(prompt, client)
                response = ollama_query.parse_response(response_text)
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
    output_path = f'{GPT_MODEL}_responses.csv'
    total, correct = process_csv_gpt(input_path, output_path)
    print(f'Finished querying {GPT_MODEL}, total examples {total}, correct predictions {correct}, accuracy {(correct/total):.4f}')
