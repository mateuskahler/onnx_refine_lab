
import json
from groq import Groq

USE_SAMPLE_DATA = True
MODEL_NAME = "llama-3.1-8b-instant"


# decide between sample data and real/local data
if USE_SAMPLE_DATA:
    prompt_file_path = "sample_system_prompt.txt"
    data_file_path = "sample_training_data.jsonl"
else:
    prompt_file_path = "system_prompt.txt"
    data_file_path = "training_data.jsonl"


def get_api_key() -> str:
    # adapt to however you want to retrieve the API key
    import keyring

    api_key = keyring.get_password("groqAPIKey", "sample_user")
    if not api_key:
        raise ValueError("API key not found in keyring.")
    return api_key


def create_groq_client() -> Groq:
    return Groq(api_key=get_api_key())


def load_system_prompt(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def load_training_data(file_path: str) -> list[dict]:
    import json
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


test_content = load_system_prompt(prompt_file_path)
first_few_lines = "\n".join([json.dumps(line)
                            for line in load_training_data(data_file_path)[:3]])


# Make the chat completion request
client = create_groq_client()
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": load_system_prompt(prompt_file_path),
        },
        {
            "role": "user",
            "content": first_few_lines,
        },
    ],
    temperature=0.0,
    # Generous headroom for 50 lines of (the expected) JSON output
    max_tokens=1500,
    seed=1
)

# Print the response text
print(response.choices[0].message.content)
