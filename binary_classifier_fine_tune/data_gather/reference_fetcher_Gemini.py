import json
from google import genai
import time

USE_SAMPLE_DATA = True
MODEL_NAME = "gemini-3.5-flash-lite"
BATCH_SIZE = 50
INTERVAL_BETWEEN_BATCHES = 15 # (seconds) for the free tier users like me :')
OUTPUT_FILE_NAME = "model_responses.jsonl"


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

    api_key = keyring.get_password("geminiAPIKey", "sample_user")
    if not api_key:
        raise ValueError("API key not found in keyring.")
    return api_key


def create_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())


def load_system_prompt(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def load_training_data(file_path: str) -> list[dict]:
    import json
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def training_data_batcher(training_data: list[dict], batch_size=50):
    """Yield successive n-sized chunks from data as a string of JSON lines."""
    for i in range(0, len(training_data), batch_size):
        batch_data = training_data[i: i + batch_size]
        data_as_a_string = "\n".join([json.dumps(line) for line in batch_data])
        yield data_as_a_string


def make_single_request(client: genai.Client, system_prompt: str, training_data: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=BATCH_SIZE * 30,
            seed=1,
        ),
        contents=training_data
    )
    result_text = response.text

    if not result_text:
        raise ValueError("No response text received from the model.")

    return result_text


def append_answers_to_file(answer: str, output_file_path: str):
    with open(output_file_path, "a", encoding="utf-8") as file:
        file.write(answer + "\n")


def main():
    system_prompt = load_system_prompt(prompt_file_path)
    training_data = load_training_data(data_file_path)
    training_batches = training_data_batcher(training_data, BATCH_SIZE)
    ai_client = create_client()

    for batch_index, batch in enumerate(training_batches):
        try:
            print(f"Processing batch [{batch_index}]...")
            response = make_single_request(ai_client, system_prompt, batch)
            time.sleep(INTERVAL_BETWEEN_BATCHES)

        except Exception as e:
            print(f"Error processing batch {batch_index}: {e}")
        else:
            append_answers_to_file(response, OUTPUT_FILE_NAME)


if __name__ == "__main__":
    main()
