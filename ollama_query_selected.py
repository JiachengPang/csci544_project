import pandas as pd
import ollama_query as query

def select_test(input_path, ids, output_path):
    df = pd.read_csv(input_path)
    selected = df.iloc[ids]
    selected.rename(columns={'label':'true_label'}, inplace=True)
    selected.to_csv(output_path, index=True)
    print(f'Saved selected data to {output_path}')

if __name__ == '__main__':
    test_path = 'snli_test.csv'
    target_ids = [20, 242, 372, 596, 707, 730, 835, 857, 304, 356, 358, 397, 561, 833]
    selected_path = 'snli_selected.csv'
    select_test(test_path, target_ids, selected_path)

    output_path = f'{query.OLLAMA_MODEL}_responses_selected.csv'
    total, correct = query.process_csv(selected_path, output_path, query.OLLAMA_MODEL)
    print(f'Finished querying {query.OLLAMA_MODEL} on selected tests, total examples {total}, correct predictions {correct}, accuracy {(correct/total):.4f}')