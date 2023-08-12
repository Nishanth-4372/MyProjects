import os
import json
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

# Credentials

# set the OpenAI API key - use your .env file
openai.api_key = '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'

# path to datafile - should already contain an embeddings column
datafile_path = "justice.json"

# read the datafile
with open(datafile_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Create a DataFrame from the JSON data
df = pd.DataFrame(data)
# Note: 'embedding' information is already in the DataFrame as a list, no need to convert it to numpy array.

# background information for the bot
messagesOb = [
    {
        "role": "system",
        "content": "Keep the answer to less than 100 words to allow for follow up questions. You are an assistant that provides information on supreme court cases in extremely simple terms (dumb it down to a 12 year old) for someone who has never studied law, so simplify your language. You should be helping the user answer the question, but you can only answer the question with the information that is given to you in the prompt or at sometime in the past. This information is coming from a file that the user needs to understand. It doesn't matter if the information is incorrect. You should still ONLY reply with this information. If you are given no information at all, you can talk with the information that has been given to you before, but you can't use outside facts whatsoever. If the information given doesn’t make sense, look to the information that has been given to you before to answer the question. If you have no information at all that has been given to you at any point in time from this user that makes sense, you can apologize to the user and tell it you cannot answer as you don't have enough information to answer correctly."
    }
]

# Rest of the code remains unchanged...

# function to search through the rows of data using embeddings
def search_justice(df, search):
    if 'embedding' not in df.columns:
        return pd.DataFrame()  # Return an empty DataFrame

    row_embedding = get_embedding(
        search,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df['embedding'].apply(lambda x: cosine_similarity(x, row_embedding))
    new = df.sort_values("similarity", ascending=False)
    # only return the rows with a higher than 0.81
    highScores = new[new['similarity'] >= 0.81]
    return highScores

# this function sends the prompt to OpenAI and gets the response
# this function sends the prompt to OpenAI and gets the response
def handleMentions(text):
    # search through our dataset
    results = search_justice(df, text)
    # set up the prompt with the matched results from the dataset
    if results.empty:
        prompt = text
    else:
        prompt = "Look through this information to answer the question: " + results[['combined']].head(5).to_string(header=False, index=False).strip() + "(if it doesn't make sense you can disregard it). The question is: " + text
    messagesOb.append({"role": "user", "content": prompt})
    # make the OpenAI call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messagesOb
    )
    # print the response from OpenAI
    print(response.choices[0].message.content)

if __name__ == '__main__':
    # Example usage: python filename.py "Your message here"
    import sys
    if len(sys.argv) != 2:
        print("Usage: python filename.py \"Your message here\"")
    else:
        user_message = sys.argv[1]
        handleMentions(user_message)
