import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from typing import List
import CustomerPromptService
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

app = FastAPI()
ip_addresses = {}

# Request format
class ChatMessage(BaseModel):
    role: str
    content: str

class RecommendationRequest(BaseModel):
    customer_query: str
    context: List[ChatMessage]

# Set up rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS middleware
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow credentials (e.g., cookies)
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")


# Initialize OpenAI LLM
llm = ChatOpenAI(
    api_key=openai_api_key, 
    temperature=0.3,
    model="gpt-4o",
    streaming=False)


# Create the prompt template
prompt = PromptTemplate(
    input_variables=["customer_query", "shops", "context"],
    template="""
    You are Santa's little helper, who's job is to recommend christmas gifts for users based on their query.
    
    Here is the previous messages (ignore the list if it is empty):
    ---
    {context}
    ---

    Here is the customer request:
    --- 
    {customer_query}
    ---

    You can only recommend gifts that available for purchase at the shopping mall "Galleriet" located in Bergen, Norway. 
    Here is a list of the shops that Galleriet has, you can ONLY recommend gifts from tehse shops:
    ---
    {shops}
    ---
    
    Respond in norwegian, in a Santa Claus-ish way, and as if you're talking directly with the customer.
    You can only recommend three gifts.
    """
)


# Endpoint for getting relevant products
@app.post("/")
@limiter.limit("100/day") # 100 request per day is max, mostly as a security measure
async def get_recommendations(request: Request, requestJson: RecommendationRequest):
    
    # Format customer query
    sanitized_query = CustomerPromptService.sanitize_input(requestJson.customer_query)
    if sanitized_query == "Invalid input detected.":
        return sanitized_query
    
    # Translate customer query to english, currently testing without this step
    #english_query = CustomerPromptService.translate_to_english(sanitized_query)
    english_query = sanitized_query
    
    # List all shops
    shops = """ Accessorize, Adam og Eva, Apotek 1, Boys of Europe, Clas Ohlson, Companys, Cubus, 
                Deguy, Delicatessen, Dogman, Emil, Fredrik & Louisa, Freequent, Gullsmed A. Lohne, 
                Gullsmed J. Gjertsen, H&M, Hjertholm, Infinity, Juice by Sportsgalleriet, Kitch'n, 
                Konfekt-galleriet, Kremmerhuset, Lakkbar, Lakrids By Bülow, Lerøy Helsekost Life, 
                Lerøy Mat, Levi's Store, Lindex, Livid, Maanesten, Match, Monki, Morland Barbers, 
                Newbie, Nille, Noe Noa, Norli Bokhandel, Optikus MON, Panduro, Rituals, Skoringen, 
                Sørensen Tobakk, Søstrene Grene, Telenor-butikken, Telia, The Body Shop, Think HairWear, 
                Tilbords, Urmaker J. Gjertsen, Vila, VITA, Wahwah, Zizzi"""
    
    # Get context
    context = requestJson.context

    # Format the prompt with the customer query and product list
    formatted_prompt = prompt.format(customer_query=english_query, shops=shops, context=context)

    print(f"Formatted prompt: {formatted_prompt}")
    
    # Pass the formatted prompt to the LLM
    response = llm.invoke(formatted_prompt)
    
    return response.content


# http://127.0.0.1:8000/docs#
# start application: 
"uvicorn Recommender:app --reload" 

# Pricing
"https://platform.openai.com/settings/organization/billing/overview"
