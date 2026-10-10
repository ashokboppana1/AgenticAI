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
Consider the following mathematical expression: 

2 + 2 * 0 + 2   
Return only the final numerical answer.
''' 

try: 
    response = client.responses.create(
        model = 'gpt-3.5-turbo',
        input = prompt
    )

    print('Response via Zero-Shot Prompting')
    print(response.output_text)
except Exception as e:   
    print(f'Error while calling the OpenAI API: {e}')