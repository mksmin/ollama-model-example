from pprint import pprint

import requests

API_KEY = "EMPTY"
BASE_URL = "http://localhost:11223/v1"
URL = f"{BASE_URL}/chat/completions"

# MODEL = "gpt-oss:20b"
# MODEL = "evilfreelancer/o1_gigachat:20b"
MODEL = "gemma3:4b"

headers = {
    "Authorization": f"Bearer {API_KEY}",
}


def main():
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Answer questions clearly an concisely. "
            "If u dont know the answer, u have to say so."
            "Otherwise provide concise answer, dont add any extra info that user didn't ask for "
        ),
    }
    user_message = {
        "role": "user",
        # "content": "What is the capital of Russia? Also give one the most important fact about the capital",
        "content": (
            "Context: square is green, triangle is red, circle is blue, hexagon is yellow, circle is purple."
            "Question: what color is circle?"
        ),
    }
    data = {
        "model": MODEL,
        "messages": [
            system_message,
            user_message,
        ],
    }

    response = requests.post(
        URL,
        json=data,
        headers=headers,
    )
    if response.ok:
        print(response.json()["choices"][0]["message"]["content"])
    else:
        print(response)
        print(response.text)


if __name__ == "__main__":
    main()
