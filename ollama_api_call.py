import requests
import json


url = "http://localhost:11434/api/chat"

premise = "A large crowd is gathered outside and one woman is yelling."

hypothesis = "A group of protestors engages with a woman inside an auditorium."

dataset_label = "neutral"

prompt = f"""What is the logical relationship between the following premise and hypothesis (one of entailment, neutral, or contradiction)? Your answer should strictly follow the given example format. For example:

Question:
Premise: A man in a tank top fixing himself a hotdog.
Hypothesis: The child was happy.

Answer: {{
    "relationship": "neutral",
    "reason": "There's no direct link suggesting that the man’s action would make the child happy, nor is there any contradiction between the two. They describe different subjects and unrelated situations. Hence, they are neutral."
}}

You are asked to answer the following question:

Premise: {premise} 
Hypothesis: {hypothesis}

Answer:"""


data = {
    "model": "llama3:latest",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False,
}


headers = {"Content-Type": "application/json"}


response = requests.post(url, headers=headers, data=json.dumps(data))


if response.status_code == 200:
    try:
        result = response.json()
        result = result["message"]
        result = result["content"]
        print(
            f"\nPremise: {premise}\nHypothesis: {hypothesis}\nLabel in dataset: {dataset_label}\n\nLlama-3.1-8B response:\n{result}\n"
        )
    except Exception as e:
        print(f"Llama response a JSON object, but could not parse it: {e}")
        print(response.text)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
