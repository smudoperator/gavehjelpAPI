import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Load your CSV file
df = pd.read_csv('norske_produkter.csv')

# Compute embeddings for each description
df['embedding'] = df['description'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

# Convert embeddings to string representation
df['embedding'] = df['embedding'].apply(lambda x: np.array(x).tolist())

# Save to CSV
df.to_csv('embedded_products.csv', index=False)