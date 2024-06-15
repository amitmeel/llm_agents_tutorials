
from dotenv import load_dotenv
load_dotenv()
import os 
import pandas as pd

from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

model = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gpt-4-1106",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

#load the dataset
df = pd.read_csv("./data/all-states-history.csv").fillna(value = 0)

# prepare langchain dataframe agent
agent = create_pandas_dataframe_agent(llm=model,df=df,verbose=True)

agent.invoke("how many rows are there?")