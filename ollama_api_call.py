import requests
import json


url = "http://localhost:11434/api/generate"

premise = "A man, woman, and child get their picture taken in front of the mountains."

hypothesis = "A family on vacation is posing."

dataset_label = "entailment"

prompt = f"""What is the logical relationship between the following premise and hypothesis (one of entailment, neutral, or contradiction)? Your answer should strictly follow the given example format and should contains only the answer part. 

Entailment means the hypothesis if definitely a true description of the premise;

Contradiction means the hypothesis is definitely false given the premise;

Neutral means the hypothesis is might be a true description of the premise, but there is no direct evidence to support it.

For example:

Question:
Premise: A man in a tank top fixing himself a hotdog.
Hypothesis: The child was happy.

Answer: {{"relationship": "neutral","reason": "There's no direct link suggesting that the man’s action would make the child happy, nor is there any contradiction between the two. They describe different subjects and unrelated situations. Hence, they are neutral."}}

You are asked to answer the following question:

Premise: {premise} 
Hypothesis: {hypothesis}

Answer:"""


data = {
    "model": "llama3:latest",
    "prompt": prompt,
    "stream": False,
}


headers = {"Content-Type": "application/json"}


response = requests.post(url, headers=headers, data=json.dumps(data))


if response.status_code == 200:
    try:
        result = response.json()
        result = result["response"]
        result = json.dumps(json.loads(result), indent=4)
        print(
            f"\nPremise: {premise}\nHypothesis: {hypothesis}\nLabel in dataset: {dataset_label}\n\nLlama-3.1-8B response:\n{result}\n"
        )
    except Exception as e:
        print(f"Llama response a JSON object, but could not parse it: {e}")
        print(response.text)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
