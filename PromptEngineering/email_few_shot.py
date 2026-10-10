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
# Few-Shot Prompting 
# ----------------------------
system_prompt = '''
You are an intelligent email classification assistant.

Classify each email into exactly ONE of these categories:

- **Complaint:** Dissatisfaction, a reported problem, or a request to resolve an issue.
- **Inquiry:** A question or request for information.
- **Feedback:** Suggestions, opinions, appreciation, or constructive comments.
- **Spam:** Unsolicited, deceptive, suspicious, or irrelevant promotional messages.

### Classification Rules
- Identify the email's primary intent.
- Select exactly one category.
- Do not copy a category from the examples without analyzing the new email.
- Return only the required output format.

### Output Format
Category: [Complaint/Inquiry/Feedback/Spam]
Reason: [One short sentence explaining the classification]
### Few-Shot Examples
few_shot_examples = [
{ 
   role: "user",
   content: "My order arrived damaged, and nobody has responded to my support request."
},
{
   role: "assistant",
   content: "Category: Complaint
   Reason: The sender is expressing dissatisfaction and reporting a problem with their order."
},
{
   role: "user",
   content: "Can you provide the specifications for the new laptop model?"
},
{
   role: "assistant",
   content: "Category: Inquiry
   Reason: The sender is asking for information about a product."
},
{
   role: "user",
   content: "I love the new features in your app! Keep up the great work."
},
{
   role: "assistant",
   content: "Category: Feedback
   Reason: The sender is providing positive feedback and appreciation."
},
{
   role: "user",
   content: "Congratulations! You've won a free vacation. Click here to claim your prize."
},
{
   role: "assistant",
   content: "Category: Spam
   Reason: The email contains unsolicited and deceptive promotional content."
}
    
]


'''

user_query = input("Enter the email text to classify: ")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

try:
    response = client.responses.create(
        model = 'gpt-4',
        input = messages
    )

    print('Response via Few-Shot Prompting')
    print(response.choices[0].message.content)
except Exception as e:
    print(f'Error while calling the OpenAI API: {e}')