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
    test_path = 'snli_test.csv'
    target_ids = [20, 242, 372, 596, 707, 730, 835, 857, 304, 356, 358, 397, 561, 833, 1253, 1350, 1405, 1561, 1571, 1598, 1677, 1695, 1779, 1780, 1841, 1915, 1927, 1964, 2031, 2152, 2160, 2220, 2225, 2357, 2450, 2507, 2637, 2700, 2701, 2801, 2819, 2885, 2915, 3152, 3153, 3155, 3166, 3211, 3250, 3271, 3329, 3346, 3429, 3466, 3494, 3520, 3701, 3733, 3813, 3829, 3876, 3884, 3890, 3918, 3991, 3995, 4000, 4069, 4086, 4087, 4153, 4278, 4325, 4396, 4439, 4444, 4464, 4498, 4529, 4534, 4592, 4624, 4689, 4743, 4782, 4798, 4799, 4803, 4887, 4922, 4966, 4973, 5101, 5159, 5170, 5196, 5205, 5208, 5233, 5239, 5299, 5300, 5374, 5464, 5524, 5629, 5647, 5779, 5849, 5959, 5991, 6003, 6130, 6158, 6184, 6220, 6322, 6344, 6401, 6552, 6562, 6566, 6617, 6763, 7026, 7215, 7223, 7242, 7289, 7332, 7355, 7416, 7440, 7522, 7539, 7562, 7586, 7705, 7711, 7782, 7827, 7915, 7972, 8172, 8259, 8275, 8340, 8405, 8550, 8719, 8822, 8852, 8977, 8986, 8986, 9078, 9186, 9269, 9270, 9402, 9410, 9439, 9479, 9488, 9499, 9606, 9620, 9660, 9686, 9730, 9800]
    selected_path = 'snli_selected.csv'
    # select_test(test_path, target_ids, selected_path)

    # model = gpt_query.GPT_MODEL
    model = ollama_query.OLLAMA_MODEL

    output_path = f'{model}_responses_selected.csv'
    # total, correct = gpt_query.process_csv_gpt(selected_path, output_path, model)
    total, correct = ollama_query.process_csv(selected_path, output_path, model)
    print(f'Finished querying {model} on selected tests, total examples {total}, correct predictions {correct}, accuracy {(correct/total):.4f}')