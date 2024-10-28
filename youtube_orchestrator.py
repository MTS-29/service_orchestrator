from IPython.display import Markdown, display
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.agents import AgentType, AgentExecutor, create_react_agent, Tool, tool
from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.schema import AgentFinish

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3,
    max_output_tokens=1024,
    top_k=40,
    top_p=0.95,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# YouTube Search Tool
youtube_search = YouTubeSearchTool()

# Wikipedia Tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def wikipedia_search_tool(query: str) -> str:
    """Useful for querying Wikipedia for factual information"""
    try:
        return wikipedia.run(query)
    except Exception as e:
        return f"Error in Wikipedia search: {str(e)}"

@tool
def youtube_search_tool(query: str) -> str:
    """search for youtube videos associated with a person. the input to this tool should
        be a comma separated list, the first part contains a person name and the second a number that
        is the maximum number of video results to return aka num_results. the second part is optional"""
    try:
        return youtube_search.run(query, 5)
    except Exception as e:
        return f"Error in Youtube search: {str(e)}"

tools = [wikipedia_search_tool, youtube_search_tool]

def create_llm_agent():
    """
    Create and return an LLM agent with the defined tools and prompt template.

    Returns:
        AgentExecutor: The configured agent executor

    This function:
    1. Defines a template for the agent's prompt, including instructions and format.
    2. Creates a PromptTemplate from this template.
    3. Creates a ReAct agent using the LLM, tools, and prompt.
    4. Returns an AgentExecutor that can run this agent with verbose output and a max of 10 iterations.
    """

    template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    When using youtube_search tool, you must provide maximum 5 videos.

    Do not iterate if you find the best answer in first few iterations.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 3 times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question including links of youtube videos if applicable

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

def process_result(result):
    """
    Process and display the result from the agent.

    Args:
        result (dict or str): The result to process

    This function handles different result formats:
    - If result is a dict with an 'output' key, it displays the output as Markdown.
    - If result is a string, it displays it as Markdown.
    - For any other format, it prints an unexpected format message.

    The use of Markdown display allows for formatted output, including links and styled text.
    """
    if isinstance(result, AgentFinish):
        return_values = result.return_values
        source = return_values
        source_type = 'return_values'
    elif isinstance(result, dict):
        return_values = result
        source = result
        source_type = 'dict'
    else:
        print(f"Debug: Unexpected result type: {type(result)}")
        return

    if 'output' in source:
        # display(Markdown(source['output']))
        print(source['output'])
    else:
        print(f"Debug: 'output' not in {source_type}")
        print(f"Debug: {source_type} keys:", source.keys())


print("Welcome to the Agentic GenAI system.")
print("You can ask complex questions, and the system will autonomously search for information from Wikipedia and Youtube.")

agent = create_llm_agent()

# Example - Create a comprehensive report on the impact of artificial intelligence in healthcare, including recent developments and future prospects. Include relevant YouTube videos.
user_input = input("What would you like to know? : ")

result = agent.invoke({"input": user_input})
process_result(result)