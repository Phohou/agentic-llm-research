from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SCRIPTS_DIR = ROOT_DIR / "scripts"


REPO_NAME_TO_DISPLAY = {
    "letta-ai/letta": "Letta",
    "microsoft/semantic-kernel": "Semantic Kernel",
    "deepset-ai/haystack": "Haystack",
    "langchain-ai/langchain": "LangChain",
    "microsoft/autogen": "AutoGen",
    "crewAIInc/crewAI": "CrewAI",
    "TransformerOptimus/SuperAGI": "SuperAGI",
    "run-llama/llama_index": "LlamaIndex",
    "FoundationAgents/MetaGPT": "MetaGPT",
}
