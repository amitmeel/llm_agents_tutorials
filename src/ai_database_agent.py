""" @copyright deeplearning.ai  """

import json
import os 
import numpy as np
import pandas as pd
from sqlalchemy import text

from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from sqlalchemy import create_engine


load_dotenv()

df = pd.read_csv("./data/all-states-history.csv").fillna(value = 0)

##   move the data to database

# Path to your SQLite database file
database_file_path = "./db/test.db"
# Create an engine to connect to the SQLite database
# SQLite only requires the path to the database file
engine = create_engine(f'sqlite:///{database_file_path}')
file_url = "./data/all-states-history.csv"
df = pd.read_csv(file_url).fillna(value = 0)
df.to_sql(
    'all_states_history',
    con=engine,
    if_exists='replace',
    index=False
)

# Prepare the sql prompt

MSSQL_AGENT_PREFIX = """
You are an agent designed to interact with a SQL database.
## Instructions: 
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.

## Tools:

"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT TOP (10) [death]
FROM covidtracking 
WHERE state = 'TX' AND date LIKE '2020%'

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought:I now know the final answer
Final Answer: There were 27437 people who died of covid in Texas in 2020.

Explanation:
I queried the `covidtracking` table for the `death` column where the state
is 'TX' and the date starts with '2020'. The query returned a list of tuples
with the number of deaths for each day in 2020. To answer the question,
I took the sum of all the deaths in the list, which is 27437.
I used the following query

```sql
SELECT [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'"
```
===> End of Example

"""

# # call the azure chat model and create an sql agent
# llm = AzureChatOpenAI(
#     openai_api_version="2023-05-15",
#     azure_deployment="gpt-4-1106",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     temperature=0, 
#     max_tokens=500
# )

# db = SQLDatabase.from_uri(f'sqlite:///{database_file_path}')
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# QUESTION = """How may patients were hospitalized during October 2020
# in New York, and nationwide as the total of all states?
# Use the hospitalizedIncrease column
# """

# agent_executor_SQL = create_sql_agent(
#     prefix=MSSQL_AGENT_PREFIX,
#     format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
#     llm=llm,
#     toolkit=toolkit,
#     top_k=30,
#     verbose=True
# )


# # invoke the sql model
# agent_executor_SQL.invoke(QUESTION)


client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  api_version="2023-05-15"
)


def get_hospitalized_increase_for_state_on_date(state_abbr, specific_date):
    try:
        query = f"""
        SELECT date, hospitalizedIncrease
        FROM all_states_history
        WHERE state = '{state_abbr}' AND date = '{specific_date}';
        """
        query = text(query)

        with engine.connect() as connection:
            result = pd.read_sql_query(query, connection)
        if not result.empty:
            return result.to_dict('records')[0]
        else:
            return np.nan
    except Exception as e:
        print(e)
        return np.nan
    

def get_positive_cases_for_state_on_date(state_abbr, specific_date):
    try:
        query = f"""
        SELECT date, state, positiveIncrease AS positive_cases
        FROM all_states_history
        WHERE state = '{state_abbr}' AND date = '{specific_date}';
        """
        query = text(query)

        with engine.connect() as connection:
            result = pd.read_sql_query(query, connection)
        if not result.empty:
            return result.to_dict('records')[0]
        else:
            return np.nan
    except Exception as e:
        print(e)
        return np.nan

# execute the function calling against sql database
tools_sql = [
    {
        "type": "function",
        "function": {
            "name": "get_hospitalized_increase_for_state_on_date",
            "description": """Retrieves the daily increase in
                              hospitalizations for a specific state
                              on a specific date.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "state_abbr": {
                        "type": "string",
                        "description": """The abbreviation of the state
                                          (e.g., 'NY', 'CA')."""
                    },
                    "specific_date": {
                        "type": "string",
                        "description": """The specific date for
                                          the query in 'YYYY-MM-DD'
                                          format."""
                    }
                },
                "required": ["state_abbr", "specific_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_positive_cases_for_state_on_date",
            "description": """Retrieves the daily increase in 
                              positive cases for a specific state
                              on a specific date.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "state_abbr": {
                        "type": "string",
                        "description": """The abbreviation of the 
                                          state (e.g., 'NY', 'CA')."""
                    },
                    "specific_date": {
                        "type": "string",
                        "description": """The specific date for the
                                          query in 'YYYY-MM-DD'
                                          format."""
                    }
                },
                "required": ["state_abbr", "specific_date"]
            }
        }
    }
]

messages = [
    {"role": "user",
     "content": """ how many hospitalized people we had in Alaska
                    the 2021-03-05?"""
    }
]



response = client.chat.completions.create(
    model="gpt-4-1106",
    messages=messages,
    tools=tools_sql,
    tool_choice="auto",
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

if tool_calls:
    print (tool_calls)
    
    available_functions = {
        "get_positive_cases_for_state_on_date": get_positive_cases_for_state_on_date,
        "get_hospitalized_increase_for_state_on_date":get_hospitalized_increase_for_state_on_date
    }  
    messages.append(response_message)  
   
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            state_abbr=function_args.get("state_abbr"),
            specific_date=function_args.get("specific_date"),
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            }
        ) 
    print(messages)

second_response = client.chat.completions.create(
            model="gpt-4-1106",
            messages=messages,
        )
print (second_response)