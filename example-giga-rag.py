from pathlib import Path
from pprint import pprint

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = Path(__file__).parent.resolve()
FILE_PATH = BASE_DIR / "data" / "rich.txt"
OPENAI_API_KEY = "EMPTY"
BASE_URL = "http://localhost:11223/v1"
# OPENAI_MODEL = "evilfreelancer/o1_gigachat:20b"
OPENAI_MODEL = "gemma3:4b"

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

embeddings = HuggingFaceEmbeddings(
    model_name="ai-forever/ru-en-RoSBERTa",
)

redis_config = RedisConfig(
    index_name="skills",
    redis_url="redis://localhost:6381",
)

vector_store = RedisVectorStore(
    config=redis_config,
    embeddings=embeddings,
)

loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

documents = text_splitter.split_documents(documents)
for d in documents:
    d.page_content = "search_document: " + d.page_content
ids = vector_store.add_documents(documents)
print(f"Indexed: {len(ids)} docs")

dim = len(embeddings.embed_query("test"))
print("Embedding dim:", dim)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    base_url=BASE_URL,
)


system_message_text_tpl = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "keep the answer concise.  "
    "Try to give short but succinct answers. If you can fit in 4 sentences, that would be great. "
    "Your answer must be in Russian. "
    "If context is enough, write 4-5 short sentences (short retelling)"
    "Context: {context}"
)

human_message_text_tpl = "{input}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_message_text_tpl),
        ("human", human_message_text_tpl),
    ]
)


def format_docs(input_docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in input_docs)


qna_chain = (
    {
        "context": vector_store.as_retriever(search_kwargs={"k": 12}) | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt_template
    | llm
)

if __name__ == "__main__":
    print("Введите вопрос (или 'exit' для выхода):")
    while True:
        question = input("> ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        q = "search_query: " + question
        results = vector_store.similarity_search_with_score(q, k=12)
        print(f"Retrieved: {len(results)} chunks")
        # for i, (doc, score) in enumerate(results, 1):
        #     print(doc.page_content[:50])  # обрезка для консоли
        answer = qna_chain.invoke(q)
        pprint(answer.content, indent=2, width=60)
