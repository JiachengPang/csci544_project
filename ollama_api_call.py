import requests
import json


url = "http://localhost:11434/api/generate"

premise = "A man, woman, and child get their picture taken in front of the mountains."

hypothesis = "A family on vacation is posing."

dataset_label = "entailment"

prompt = f"""
The logical relationship between the following premise and hypothesis is defined as one of the following:

neutral: the hypothesis might be a true description of the premise, but there is no direct evidence to support it, you can neither accept nor reject the hypothesis based on the premise;
contradiction: the hypothesis is definitely false given the premise, you can reject the hypothesis based on the premise;
entailment: the hypothesis is definitely true given the premise, you can accept the hypothesis based on the premise.

Examples:
    Question:
    Premise: A boy is drinking out of a water fountain shaped like a woman.
    Hypothesis: A sculptor takes a drink from a fountain that he made that looks like his girlfriend.

    Answer:
    {{"cot": "1. Premise Analysis: The premise states 'A boy is drinking out of a water fountain shaped like a woman.' It describes a boy drinking from a fountain, with no information about who made the fountain or the boy's relationship to it. 2. Hypothesis Analysis: The hypothesis says 'A sculptor takes a drink from a fountain that he made that looks like his girlfriend.' It introduces two additional details: the person drinking is a sculptor, and the fountain was made by him to resemble his girlfriend. 3. Comparing the Two: The premise provides no direct information about the boy being a sculptor, nor that he created the fountain or that it looks like his girlfriend. While this scenario might be possible, the premise does not offer any evidence for the hypothesis. 4. Conclusion: Since the hypothesis might be true but there is no direct evidence to support it, the relationship is neutral.", "relationship": "neutral"}}

    Question:
    Premise: A boy is drinking out of a water fountain shaped like a woman.
    Hypothesis: A man is drinking lemonade from a glass.

    Answer:
    {{"cot": "1. Premise Analysis: The premise states that 'A boy is drinking out of a water fountain shaped like a woman.' This provides specific information that a boy is drinking from a water fountain, not from a glass or any other container, and it mentions the type of drink is water. 2. Hypothesis Analysis: The hypothesis says 'A man is drinking lemonade from a glass.' This introduces new information: the drink is lemonade from a glass. 3. Comparing the Two: The premise clearly states the that the boy is drinking from a water fountain, not a glass, and it confirms the drink is water. 4. Conclusion: The hypothesis directly contradicts the premise because the drink, and the container are both different. Therefore, the relationship is contradiction.", "relationship": "contradiction"}}

    Question:
    Premise: A boy is drinking out of a water fountain shaped like a woman.
    Hypothesis: A male is getting a drink of water.

    Answer:
    {{"cot": "1. Premise Analysis: The premise states that 'A boy is drinking out of a water fountain shaped like a woman.' This clearly describes a male (the boy) who is drinking water. 2. Hypothesis Analysis: The hypothesis says 'A male is getting a drink of water.' This aligns directly with the premise, as the boy is male and is drinking water. 3. Comparing the Two: The premise directly supports the hypothesis, as the action and the subject (a male drinking water) are explicitly mentioned. 4. Conclusion: The hypothesis is definitely true given the premise, so the relationship is entailment.", "relationship": "entailment"}}

What is the logical relationship between the following premise and hypothesis? Your answer should: 1. strictly follow the standard parseable JSON format: {{"cot": <your chain-of-though>, "relationship": <your answer>}}; 2. based solely on the provided information, without any common sense knowledge or external information; 3. contain only the answer part, avoid using stuff like 'Let's analyze the premise and hypothesis:' or 'Here is my answer:'.

Question:
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
        print(response.json()["response"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
