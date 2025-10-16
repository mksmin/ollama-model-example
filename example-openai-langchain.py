from pprint import pprint
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

OPEN_AI_API_KEY = "EMPTY"
OPEN_AI_BASE_URL = "http://localhost:11223/v1"
MODEL = "gemma3:4b"

llm = ChatOpenAI(
    model_name=MODEL,
    api_key=OPEN_AI_API_KEY,
    base_url=OPEN_AI_BASE_URL,
)


class ResponseFormat(BaseModel):
    answer: str = Field(description="The answer to the user's question")


class FastapiIdea(BaseModel):
    """
    This model describes short fastapi apps ideas for beginners.
    This app should include an uv, pydantic schemas and black formatting
    """

    title: str = Field(description="Title of the app")
    short_description: str = Field(
        description="The short description of the app: should contain a couple of sentences"
    )
    details: str = Field(
        description="The details of the app: should contain a couple of sentences about functions of app"
    )
    # topics: list[str] = Field(
    #     description="The topics of the app: should be a list of strings. From 8 to 10 topics for the app using"
    # )


class FastapiIdeasResponse(BaseModel):
    ideas: list[FastapiIdea]


system_prompt = """
You are a helpful assistant. 
Answer questions clearly an concisely. 
If u dont know the answer, u have to say so.
Otherwise provide concise answer, dont add any extra info that user didn't ask for 
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{user_input}"),
    ],
)

prepared_prompt = prompt_template.invoke(
    "What is the capital of Russia? Also give me one the most important fact about it"
)

prompt_template_with_context = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "Context: {context}\nQuery: {user_input}"),
    ],
)
human_context = "square is green, triangle is red, circle is blue, hexagon is yellow, circle is purple."
human_prompt = "What color is square"

prepared_prompt_with_context = prompt_template_with_context.invoke(
    {
        "context": human_context,
        "user_input": human_prompt,
    }
)

chain = prompt_template_with_context | llm

model_with_struct = llm.with_structured_output(ResponseFormat)
chain_with_struct = prompt_template_with_context | model_with_struct

model_for_video_ideas = llm.with_structured_output(FastapiIdeasResponse)
fastapi_ideas_chain = prompt_template_with_context | model_for_video_ideas
ideas_context = (
    "Senior python developer, every days affirmations, tasks tracker, 10 ideas"
)
ideas_query = "Please create fastapi ideas for beginners about using LLM un Python web applications using langchain. Please give at least 10 ideas "

if __name__ == "__main__":
    # answer = llm.invoke(prepared_prompt)
    # answer = llm.invoke(prepared_prompt_with_context)
    # answer = chain.invoke(
    #     {
    #         "context": human_context,
    #         "user_input": human_prompt,
    #     }
    # )
    # pprint(answer, indent=4)
    # print(answer.content)

    answer = chain_with_struct.invoke(
        {
            "context": human_context,
            "user_input": human_prompt,
        }
    )
    print(answer.answer)

    response = fastapi_ideas_chain.invoke(
        {
            "context": ideas_context,
            "user_input": ideas_query,
        }
    )
    for idea in response.ideas:
        print()
        print(idea.title)
        print(idea.short_description)
        print(idea.details)
