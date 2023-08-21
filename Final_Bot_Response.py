#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gpt_2_simple as gpt2
import re

def generate_response(user_input, sess):
    max_tokens = 200  # Maximum number of tokens to generate
    temperature = 0.7  # Controls the randomness of the response

    response = gpt2.generate(sess, prefix=user_input, length=max_tokens, temperature=temperature, return_as_list=True)[0]
    sentences = re.split(r'(?<=[.!?])\s', response, maxsplit=6)  # Extract two sentences from the response

    return ' '.join(sentences[:6])  # Join the first two sentences into a single response

# Load the saved model
model_dir = '/Users/rusheentakhtani/checkpoint/run1'  # Path to the saved model directory
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_dir=model_dir)

user_input = ""
bot_response = ""

while True:
    user_input = input("User: ").strip().lower()  # Remove leading/trailing spaces and convert to lowercase
    bot_response = generate_response(user_input, sess)

    # Check if the generated response starts with the input question
    if bot_response.lower().startswith(user_input.lower()):
        bot_response = bot_response[len(user_input):].lstrip('?.,!')  # Remove the question from the response

    print("Bot:", bot_response)


# In[ ]:




