from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool

# settings
Settings.llm = OpenAI(model="gpt-4o-mini",temperature=0)

# function tools
def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# rag pipeline
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# response = query_engine.query("What was the total amount of the 2023 Canadian federal budget?")
# print(response)

# rag pipeline as a tool
budget_tool = QueryEngineTool.from_defaults(
    query_engine, 
    name="相关资料",
    description="相关资料"
)

agent = ReActAgent.from_tools([multiply_tool, add_tool, budget_tool], verbose=True)

response = agent.chat("在食品安全国家标准的肉与肉制品采样和检样处理中，对于冷冻样品，在40℃时最多能解冻多少分钟，将它乘以300是多少个小时，逐步思考回答")

print(response)