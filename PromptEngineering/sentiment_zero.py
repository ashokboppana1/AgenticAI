import os
from dotenv import load_dotenv
from openai import OpenAI

# load environment variables from .env
load_dotenv()

# Validate the API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_APU_KEY is not in the .env file")

client = OpenAI()
# ----------------------------
# Zero-Shot Prompting   
# ----------------------------
prompt = '''
You are an intelligent sentiment analysis assistant.
Your task is to classify the sentiment of the given text into exactly ONE of the following categories:

Positive - The text expresses a favorable opinion, satisfaction, or happiness about a product, service, experience, or situation.
Negative - The text expresses dissatisfaction, disappointment, or unhappiness about a product, service, experience, or situation.
Neutral - The text expresses neither a strong positive nor negative sentiment; it is factual, objective, or indifferent.

Classification Rules:
Select only one category.
Choose the category that best represents the text's primary sentiment.
Return the result in the specified format.
Output Format:
Category: [Positive/Negative/Neutral]
Text:
{input_text}


'''

try:
    response = client.responses.create(
        model = 'gpt-4',
        input = prompt.format(input_text="I recently bought a new phone, and I am extremely satisfied with its performance and features.") 
    )

    print('Response via Zero-Shot Prompting')
    print(response.output_text)
except Exception as e:
    print(f'Error while calling the OpenAI API: {e}')