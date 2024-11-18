import pandas as pd
import ollama_query as query

total, correct = query.process_csv('snli_test_copy.csv', f'{query.OLLAMA_MODEL}_snli_responses.csv', query.OLLAMA_MODEL)
print(f'Finished querying {query.OLLAMA_MODEL} on SNLI test set, total examples {total}, correct predictions {correct}, accuracy {(correct/total):.4f}')

df = pd.read_csv(f'{query.OLLAMA_MODEL}_snli_responses.csv')
total = len(df) - 1
correct = (df['true_label'] == df['response']).sum()
print(f'total {total}, correct predictions {correct}, accuracy {(correct/total):.4f}')