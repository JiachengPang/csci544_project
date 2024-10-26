import requests
import json


url = "http://localhost:11434/api/generate"

premise = "A man, woman, and child get their picture taken in front of the mountains."

hypothesis = "A family on vacation is posing."

dataset_label = "entailment"

prompt = f"""
The logical relationship between the following premise and hypothesis is defined as one of the following:
Neutral means the hypothesis might be a true description of the premise, but there is no direct evidence to support it.
Contradiction means the hypothesis is definitely false given the premise;
Entailment means the hypothesis is definitely true given the premise.

What is the logical relationship between the following premise and hypothesis? Your answer should strictly follow the standard parseable JSON format: {{"relationship": <your answer>, "cot": <your chain-of-though>}} and should contain only the answer part, avoid using stuff like 'Let's analyze the premise and hypothesis:' or 'Here is my answer:'. 

For example:

Question:
Premise: A person with a purple shirt is painting an image of a woman on a white wall.
Hypothesis: A woman paints a portrait of a person.

Answer:
{{"relationship": "neutral", "cot": "1. Premise Analysis: The premise states that 'A person with a purple shirt is painting an image of a woman on a white wall.' This provides specific information about the painter (wearing a purple shirt) and the subject being painted (a woman). 2. Hypothesis Analysis: The hypothesis says 'A woman paints a portrait of a person.' This introduces two ideas : The painter is a woman. She is painting a portrait of someone. 3. Comparing the Two: The premise does not explicitly say that the painter is a woman, nor does it confirm or deny the idea that she is painting a portrait. It only mentions the person with the purple shirt painting an image of a woman on a white wall. While it's possible that the painter could be a woman and painting a portrait, the premise doesn't provide direct evidence for this. 4. Conclusion: Since the hypothesis might be true (it's possible she's a woman painting a portrait) but there is no direct evidence provided in the premise, this makes the relationship neutral. There is no contradiction, nor an entailment."}}

You are asked to answer the following question:

Premise: {premise} 
Hypothesis: {hypothesis}

Answer:
"""


data = {
    "model": "llama3:latest",
    "prompt": prompt,
    "stream": False,
}


headers = {"Content-Type": "application/json"}


response = requests.post(url, headers=headers, data=json.dumps(data))

# print(f"{prompt=}")

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
