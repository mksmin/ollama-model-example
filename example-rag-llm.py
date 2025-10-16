import requests
from pprint import pprint
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

OPENAI_API_KEY = "EMPTY"
BASE_URL = "http://localhost:11223/v1"
OLLAMA_BASE_URL = "http://localhost:11223"
EMB_URL = f"{BASE_URL}/embeddings"

# EMB_MODEL = "hf.co/evilfreelancer/FRIDA-GGUF:latest"
EMB_MODEL = "hf.co/lmstudio-community/granite-embedding-278m-multilingual-GGUF"
# OPENAI_MODEL = "evilfreelancer/o1_gigachat:20b"
# OPENAI_MODEL = "gemma3:4b"
OPENAI_MODEL = "gpt-oss:20b"

BASE_DIR = Path(__file__).parent.resolve()
FILE_PATH = BASE_DIR / "data" / "мастер_и_маргарита.txt"

redis_config = RedisConfig(
    index_name="master_and_margarita_idx_new",
    redis_url="redis://localhost:6381",
)
embeddings = OllamaEmbeddings(
    model=EMB_MODEL,
    base_url=OLLAMA_BASE_URL,
)

vector_store = RedisVectorStore(
    config=redis_config,
    embeddings=embeddings,
)

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=BASE_URL,
)

loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

documents = text_splitter.split_documents(documents)

vector_store.add_documents(documents)

system_message_text_tpl = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentence maximum and keep the answer concise "
    "Your answer must be in Russian. "
    "Context: {context}"
)
human_message_text_tpl = "{input}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_message_text_tpl),
        ("human", human_message_text_tpl),
    ]
)
# context = None
question = "Какой был плащ у Понтия? Опиши детали"

# prompt = prompt_template.invoke(
#     {
#         "context": context,
#         "user_input": question,
#     }
# )

qna_chain = create_stuff_documents_chain(
    llm,
    prompt_template,
)
rag_chain = create_retrieval_chain(
    vector_store.as_retriever(),
    qna_chain,
)


def format_docs(input_docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in input_docs)


qna_chain_new = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt_template
    | llm
)


def generate_embeddings(text: str) -> list[float]:
    data = {
        "model": EMB_MODEL,
        "input": text,
    }
    response = requests.post(
        url=EMB_URL,
        json=data,
        headers=headers,
    )
    response.raise_for_status()
    response_data = response.json()

    pprint(response_data["data"][0]["embedding"], indent=2)


def only_output(text: str) -> str:
    import re

    m = re.search(
        r"<\s*Output\s*>\s*(.*?)\s*<\s*/\s*Output\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else text.strip()


if __name__ == "__main__":
    # answer = llm.invoke("Какой плащ у Понтия Пилата?")
    # print(only_output(answer.content))

    # vectors = vector_store.similarity_search(
    #     query="плащ",
    #     k=3,
    # )
    # for vector in vectors:
    #     pprint(vector.page_content)

    # answer = rag_chain.invoke({"input": question})
    # pprint(answer)

    answer = qna_chain_new.invoke(question)
    pprint(answer.content, indent=2)
    pprint(answer)
