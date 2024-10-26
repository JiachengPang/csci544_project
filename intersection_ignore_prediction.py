import pandas as pd

# Define the filenames of your CSV files
filenames = ['deberta_misclassified_examples.csv', 'bert_misclassified_examples.csv', 'roberta_misclassified_examples.csv', 'albert_misclassified_examples.csv']

# Read each CSV file into a pandas DataFrame
dataframes = []
for file in filenames:
    try:
        df = pd.read_csv(file, encoding='utf-8')
        dataframes.append(df.drop(columns=['predicted_label']))
        print(f"Successfully read {file} with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file {file} was not found.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file} is empty.")
        exit(1)
    except pd.errors.ParserError:
        print(f"Error: The file {file} does not appear to be in CSV format.")
        exit(1)

# Verify that all DataFrames have the same columns
base_columns = list(dataframes[0].columns)
for i, df in enumerate(dataframes[1:], start=2):
    if list(df.columns) != base_columns:
        print(f"Error: The columns in file{i} do not match the columns in file1.")
        exit(1)

# Find the intersection of all DataFrames
# Convert each DataFrame to a set of tuples for intersection
sets_of_rows = [set(tuple(row) for row in df.to_numpy()) for df in dataframes]

# Perform intersection across all sets
intersection_set = set.intersection(*sets_of_rows)
print(f"Number of intersecting rows: {len(intersection_set)}")

# If no intersection is found, inform the user
if not intersection_set:
    print("No common rows found across all files.")
    exit(0)

# Convert the intersection set back to a DataFrame
intersection_df = pd.DataFrame(list(intersection_set), columns=base_columns)

# Ensure the 'index' column is of integer type for proper sorting
if 'index' not in intersection_df.columns:
    print("Error: The 'index' column was not found in the data.")
    exit(1)

# Convert 'index' to numeric, coercing errors and dropping rows where conversion fails
intersection_df['index'] = pd.to_numeric(intersection_df['index'], errors='coerce')
before_drop = len(intersection_df)
intersection_df.dropna(subset=['index'], inplace=True)
after_drop = len(intersection_df)
if before_drop != after_drop:
    print(f"Dropped {before_drop - after_drop} rows due to non-numeric 'index' values.")

# Convert 'index' to integer type
intersection_df['index'] = intersection_df['index'].astype(int)

# Sort the DataFrame by the 'index' column in ascending order
intersection_df.sort_values(by='index', inplace=True)

# Reset the index of the DataFrame
intersection_df.reset_index(drop=True, inplace=True)

# Write the sorted intersection DataFrame to a new CSV file
output_filename = 'intersection_output_ignore_prediction.csv'
intersection_df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"The intersection has been written to '{output_filename}' sorted by the 'index' column.")
