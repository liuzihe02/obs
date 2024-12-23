import os
from phoenix.otel import register
from dotenv import load_dotenv

# langchain imports
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Add Phoenix API Key for tracing
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"

# configure the Phoenix tracer
tracer_provider = register(
    project_name="my-llm-app",  # Default is 'default'
    endpoint="https://app.phoenix.arize.com/v1/traces",
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


prompt = ChatPromptTemplate.from_template("{x} {y} {z}?").partial(x="why is", z="blue")
chain = prompt | ChatOpenAI(model_name="gpt-3.5-turbo")
chain.invoke(dict(y="apple"))
