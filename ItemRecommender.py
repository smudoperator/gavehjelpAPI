import os
import pandas as pd
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# Read the CSV file
df = pd.read_csv('products.csv')
print(df.head())

print(f"API Key: {openai_api_key}")

# openai_api_key = "fix local variables" 
llm = OpenAI(api_key=openai_api_key, temperature=0.7)

# Define a template for prompt
prompt = PromptTemplate(
    input_variables=["customer_query", "products"],
    template="""
    A customer is looking for products based on this input: {customer_query}.
    Here is a list of products:
    {products}
    
    Based on customer interests, recommend the most relevant products.
    """
)

# Create an LLMChain
chain = LLMChain(llm=llm, prompt=prompt)


def get_relevant_products(customer_query):
    # Prepare the product descriptions from the DataFrame
    product_list = "\n".join([f"{row['product_name']}: {row['description']} (${row['price']})" 
                              for _, row in df.iterrows()])
    
    # Pass the query and product list to the LLM chain
    response = chain.run(customer_query=customer_query, products=product_list)
    
    return response


# Testing 
customer_query = "I need something to help automate my home and play music."
recommendations = get_relevant_products(customer_query)
print(recommendations)


# Pricing
"https://platform.openai.com/settings/organization/billing/overview"