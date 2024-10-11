import os
from dotenv import load_dotenv
from langchain_openai import OpenAI as LangChainOpenAI

# This class deals with the customer prompt
# Sanitazation
# Translation
# Embedding



# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")


def translate_to_english(text: str) -> str:
    translation_prompt = f"Translate the following text to English:\n\n{text}"
    llm = LangChainOpenAI(api_key=openai_api_key, temperature=0.3)
    response = llm(translation_prompt)
    
    # Print the response for debug purposes
    print(response)
    
    return response.strip()



# Function for getting gender
def get_gender_from_query(query: str) -> str:
    terms_to_gender = {
        "boy": "male",
        "son": "male",
        "daughter": "female",
        "girl": "female",
        "man": "male",
        "woman": "female",
        "male": "male",
        "female": "female",
        "uncle": "male",
        "aunt": "female"
    }
    
    for term, gender in terms_to_gender.items():
        if term in query.lower():
            return gender
    
    return "unisex"


# Deal with prompt injection (should find a better way to do this)
def sanitize_input(input_text):
    blacklist = ["ignore", "system:", "forget", "instructions", "assistant:", "exit", "restart", "shut down", "delete"]
    blacklist_norwegian = ["ignorer", "system", "glem", "instrukser", "instruksjoner", "ordre"]
    blacklist += blacklist_norwegian
    for word in blacklist:
        if word.lower() in input_text.lower():
            return "Invalid input detected."
    return input_text