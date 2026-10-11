import os
from dotenv import load_dotenv
from openai import OpenAI

# load environment variables from .env
load_dotenv()

# Validate the API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_APU_KEY is not in the .env file")

client = OpenAI()

#----------------------------
# Chain-of-Thought Prompting
#----------------------------

prompt = '''
Solve the problem carefully
Provide a concise explanation and final answer.

'''
user_prompt = input("Enter your mathematical expression: ")
message = [
  {'role': "system", 'content':prompt},
  { 'role': "user", 'content':user_prompt}

]

try:
    response = client.responses.create(
        model = 'gpt-3.5-turbo',
        input = prompt
    )

    print('Response via Chain-of-Thought Prompting')
    print(response.output_text)
except Exception as e:
    print(f"An error occurred: {e}")