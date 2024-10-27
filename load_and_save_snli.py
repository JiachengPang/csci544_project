from datasets import load_dataset

# Load the SNLI dataset
snli = load_dataset("snli")

def remove_no_label(example):
    return example['label'] != -1

snli = snli.filter(remove_no_label)

# Save train, validation, and test sets to CSV files
train_df = snli['train'].to_pandas()
validation_df = snli['validation'].to_pandas()
test_df = snli['test'].to_pandas()

# File paths
train_file_path = "snli_train.csv"
validation_file_path = "snli_validation.csv"
test_file_path = "snli_test.csv"

# Save the datasets to disk
train_df.to_csv(train_file_path, index=True, index_label='index')
validation_df.to_csv(validation_file_path, index=True, index_label='index')
test_df.to_csv(test_file_path, index=True, index_label='index')

print(f"Files saved: {train_file_path}, {validation_file_path}, {test_file_path}")