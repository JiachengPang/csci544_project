from datasets import load_dataset
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import nltk

snli_dataset = load_dataset("snli")

def label_distribution(dataset=snli_dataset):
    snli_df = dataset['train'].to_pandas()

    # -1 as "wrong label"
    label_distribution = snli_df['label'].value_counts()
    label_distribution.index = label_distribution.index.map({0: 'entailment', 1: 'neutral', 2: 'contradiction', -1: 'wrong label'})

    plt.figure(figsize=(10, 6))
    ax = label_distribution.plot(kind='bar', color='skyblue')
    plt.title("SNLI Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Sample Size")

    for i, count in enumerate(label_distribution):
        ax.text(i, count + 500, str(count), ha='center', va='bottom')
        
    plt.xticks(rotation=0)

    plt.show()

def sentence_length(dataset=snli_dataset):

    snli_df = dataset['train'].to_pandas()
    filtered_df = snli_df[snli_df['label'] != -1]

    filtered_df['premise_length'] = filtered_df['premise'].apply(lambda x: len(x.split()))
    filtered_df['hypothesis_length'] = filtered_df['hypothesis'].apply(lambda x: len(x.split()))


    plt.figure(figsize=(12, 6))
    plt.hist(filtered_df['premise_length'], bins=30, alpha=0.7, color='blue', label='Premise Length')
    plt.hist(filtered_df['hypothesis_length'], bins=30, alpha=0.7, color='orange', label='Hypothesis Length')
    plt.title("Length Distribution of Premise and Hypothesis Sentences in SNLI Dataset")
    plt.xlabel("Sentence Length")
    plt.ylabel("Number of Sentence")
    plt.legend()
    plt.show()
    
def words_frequency(dataset=snli_dataset):
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    filtered_dataset = dataset["train"].filter(lambda x: x['label'] != -1)

    def preprocess_text(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words]  # 过滤停用词

    def get_label_word_frequencies(dataset, column, label_value):
        label_data = dataset.filter(lambda x: x['label'] == label_value)
        words = [word for text in label_data[column] for word in preprocess_text(text)]
        return Counter(words)

    label_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    label_word_freqs = {
        label: {
            'premise': get_label_word_frequencies(filtered_dataset, 'premise', label),
            'hypothesis': get_label_word_frequencies(filtered_dataset, 'hypothesis', label)
        } for label in label_names.keys()
    }

    plt.figure(figsize=(18, 12))

    for i, (label, freqs) in enumerate(label_word_freqs.items(), start=1):
        top_premise_words = freqs['premise'].most_common(20)
        top_hypothesis_words = freqs['hypothesis'].most_common(20)
        
        plt.subplot(3, 2, 2 * i - 1)
        plt.bar(*zip(*top_premise_words), color='skyblue')
        plt.title(f"Top 20 Words in Premise Sentences ({label_names[label]})")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, 2 * i)
        plt.bar(*zip(*top_hypothesis_words), color='lightcoral')
        plt.title(f"Top 20 Words in Hypothesis Sentences ({label_names[label]})")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # sentence_length()
    # label_distribution()
    words_frequency()



