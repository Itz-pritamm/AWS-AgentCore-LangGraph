import csv
import os
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ✅ Memory imports
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.store.base import BaseStore

# Initialize app
app = BedrockAgentCoreApp()
_ = load_dotenv()
load_dotenv()


# ================= MEMORY SETUP =================

MEMORY_ID = "coustomer_care_agent_memory-psIi7ZBfeN"   # keep simple

checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID)
store = AgentCoreMemoryStore(memory_id=MEMORY_ID)


class MemoryMiddleware(AgentMiddleware):
    def pre_model_hook(self, state: AgentState, config: RunnableConfig, *, store: BaseStore):
        actor_id = config["configurable"]["actor_id"]
        thread_id = config["configurable"]["thread_id"]

        namespace = (actor_id, thread_id)
        messages = state.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                store.put(namespace, str(uuid.uuid4()), {"message": msg})
                break

        return {"messages": messages}

    def post_model_hook(self, state, config: RunnableConfig, *, store: BaseStore):
        actor_id = config["configurable"]["actor_id"]
        thread_id = config["configurable"]["thread_id"]

        namespace = (actor_id, thread_id)
        messages = state.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                store.put(namespace, str(uuid.uuid4()), {"message": msg})
                break

        return state


# ================= DATA =================

def load_faq_csv(path: str) -> List[Document]:
    """Load FAQ data from CSV and convert to LangChain Documents."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            docs.append(Document(page_content=f"Q: {q}\nA: {a}"))
    return docs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "lauki_qna.csv")

docs = load_faq_csv(csv_path)


def simple_search(query: str, k: int = 3):
    """Basic keyword search over FAQ documents."""
    results = []
    query_lower = query.lower()

    for doc in docs:
        if query_lower in doc.page_content.lower():
            results.append(doc)

    return results[:k]


# ================= TOOLS =================

@tool
def search_faq(query: str) -> str:
    """Search FAQ entries based on a query and return top matching answers."""
    results = simple_search(query, k=3)

    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def search_detailed_faq(query: str, num_results: int = 5) -> str:
    """Search FAQ with more results for deeper context."""
    results = simple_search(query, k=num_results)

    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def reformulate_query(original_query: str, focus_aspect: str) -> str:
    """Reformulate a query to focus on a specific aspect and search again."""
    reformulated = f"{focus_aspect} related to {original_query}"
    results = simple_search(reformulated, k=3)

    if not results:
        return f"No results found for aspect: {focus_aspect}"

    context = "\n\n---\n\n".join([
        f"Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Results for '{focus_aspect}' aspect:\n\n{context}"


tools = [search_faq, search_detailed_faq, reformulate_query]


# ================= MODEL =================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


system_prompt = """You are a helpful FAQ assistant with memory.

- Remember past user interactions
- Use tools for answering
- Be clear and concise
"""


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    store=store,
    middleware=[MemoryMiddleware()]
)


# ================= ENTRYPOINT =================

@app.entrypoint
def agent_invocation(payload, context):
    """Main handler for Bedrock AgentCore runtime with memory."""

    query = payload.get("prompt", "No prompt found")

    actor_id = payload.get("actor_id", "default-user")
    thread_id = payload.get("thread_id", "default-session")

    config = {
        "configurable": {
            "thread_id": thread_id,
            "actor_id": actor_id
        }
    }

    result = agent.invoke(
        {"messages": [("human", query)]},
        config=config
    )

    return {
        "result": result['messages'][-1].content,
        "actor_id": actor_id,
        "thread_id": thread_id
    }


if __name__ == "__main__":
    app.run()