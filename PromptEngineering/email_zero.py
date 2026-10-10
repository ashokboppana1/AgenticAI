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
You are an intelligent email classification assistant.
Your task is to classify the given email into exactly ONE of the following categories:

Complaint - The sender expresses dissatisfaction, reports a problem, or complains about a product, service, or experience.

Inquiry - The sender asks a question or requests information about a product, service, order, pricing, or policy.

Feedback -The sender provides suggestions, opinions, appreciation, or constructive comments about a product or service without primarily reporting a problem.

Spam - The email contains unsolicited promotions, suspicious offers, scams, irrelevant advertisements, or deceptive content.

Classification Rules

Select only one category.

Choose the category that best represents the email's primary intent.

Do not assume an email is spam simply because it contains promotional language; consider whether it is deceptive, unsolicited, or irrelevant.

Return the result in the specified format.

Output Format
Category: [Complaint/Inquiry/Feedback/Spam]
Reason: [One short sentence explaining the classification]

Email:
{email_text}

'''

try: 
    response = client.responses.create(
        model = 'gpt-4',
        input = prompt.format(email_text="I purchased a laptop last week, but it stopped working after two days. Please resolve this issue.") 
    )

    print('Response via Zero-Shot Prompting')
    print(response.output_text)
except Exception as e:   
    print(f'Error while calling the OpenAI API: {e}')
