import pandas as pd
import ollama_query
import gpt_query

def select_test(input_path, ids, output_path):
    df = pd.read_csv(input_path)
    selected = df.iloc[ids]
    selected.rename(columns={'label':'true_label'}, inplace=True)
    selected.to_csv(output_path, index=True)
    print(f'Saved selected data to {output_path}')

if __name__ == '__main__':
    # test_path = 'snli_test.csv'
    # target_ids = [20, 242, 372, 596, 707, 730, 835, 857, 304, 356, 358, 397, 561, 833, 1253, 1350, 1405, 1561, 1571, 1598, 1677, 1695, 1779, 1780, 1841, 1915, 1927, 1964, 2031, 2152, 2160, 2220, 2225, 2357, 2450, 2507, 2637, 2700, 2701, 2801, 2819, 2885, 2915, 3152, 3153, 3155, 3166, 3211, 3250, 3271, 3329, 3346, 3429, 3466, 3494, 3520, 3701, 3733, 3813, 3829, 3876, 3884, 3890, 3918, 3991, 3995, 4000, 4069, 4086, 4087]
    selected_path = 'snli_selected.csv'
    # select_test(test_path, target_ids, selected_path)

    model = gpt_query.GPT_MODEL

    output_path = f'{model}_responses_selected.csv'
    total, correct = gpt_query.process_csv_gpt(selected_path, output_path, model)
    print(f'Finished querying {model} on selected tests, total examples {total}, correct predictions {correct}, accuracy {(correct/total):.4f}')