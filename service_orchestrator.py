import os
from langchain.agents import Tool, initialize_agent
from langchain.llms import GooglePalm  # Placeholder for Gemini via VertexAI or GooglePalm LLM
from langchain.agents import AgentType
from langchain.tools import tool
import google.generativeai as genai
import requests
from dotenv import load_dotenv
load_dotenv()


# Initialize Google Gemini (via google.generativeai)
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Define LLM for the agent (Gemini-pro)
llm = GooglePalm(model="gemini-pro")


# Helper function for API requests
def api_request(method: str, url: str, headers=None, params=None, data=None):
    """Handles API requests with error handling."""
    try:
        response = requests.request(method, url, headers=headers, params=params, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"error": str(http_err)}
    except Exception as err:
        return {"error": str(err)}


# 1. Jira Tool
@tool
def get_jira_tickets(query: str) -> str:
    """Fetches JIRA tickets based on a project key."""
    project_key = query.strip()
    JIRA_API_BASE_URL = os.getenv('JIRA_API_BASE_URL')
    JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
    
    headers = {
        'Authorization': f'Bearer {JIRA_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    url = f'{JIRA_API_BASE_URL}/rest/api/2/search'
    params = {'jql': f'project={project_key}', 'maxResults': 10}
    
    response = api_request('GET', url, headers=headers, params=params)
    
    if 'issues' in response:
        return "\n".join([f"Ticket: {ticket['key']}, Summary: {ticket['fields']['summary']}" 
                          for ticket in response['issues']])
    return "No tickets found or an error occurred."


# 2. Jira Help Tool
@tool
def jira_help(query: str) -> str:
    """Fetches help articles from the JIRA help system."""
    JIRA_HELP_API_URL = os.getenv('JIRA_HELP_API_URL')
    JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
    
    headers = {
        'Authorization': f'Bearer {JIRA_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    url = f'{JIRA_HELP_API_URL}/search'
    data = {'query': query}
    
    response = api_request('POST', url, headers=headers, data=data)
    
    if 'articles' in response:
        return "\n".join([f"Article: {article['title']}, Link: {article['link']}"
                          for article in response['articles']])
    return "No help articles found or an error occurred."


# 3. Knowledge Base Tool
@tool
def get_knowledge_base_articles(query: str) -> str:
    """Fetches knowledge base articles on a specific topic."""
    KNOWLEDGE_BASE_API_URL = os.getenv('KNOWLEDGE_BASE_API_URL')
    JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
    
    headers = {
        'Authorization': f'Bearer {JIRA_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    url = f'{KNOWLEDGE_BASE_API_URL}/articles'
    params = {'topic': query.strip()}
    
    response = api_request('GET', url, headers=headers, params=params)
    
    if 'articles' in response:
        return "\n".join([f"Article: {article['title']}, URL: {article['url']}"
                          for article in response['articles']])
    return "No knowledge base articles found or an error occurred."


# Define the tools list for the agent
tools = [
    Tool(name="Get JIRA Tickets", func=get_jira_tickets, description="Use to retrieve JIRA tickets"),
    Tool(name="JIRA Help", func=jira_help, description="Use to retrieve help articles from JIRA help system"),
    Tool(name="Knowledge Base", func=get_knowledge_base_articles, description="Use to get knowledge base articles")
]

# Initialize the agent using the tools and the Gemini model
agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Zero-shot reasoning
    llm=llm,
    verbose=True
)

# Run the agent for a user query
def agent_run():
    while True:
        query = input("Enter your query: ")
        result = agent.run(query)
        print(result)

# Start agent
if __name__ == "__main__":
    agent_run()
