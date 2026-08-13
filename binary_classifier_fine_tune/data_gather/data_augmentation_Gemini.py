import json
from google import genai

MODEL_NAME = "gemini-3.5-flash-lite"
BATCH_SIZE = 75
NUMBER_OF_BATCHES = 2
OUTPUT_FILE_NAME = "model_synthetic.jsonl"


prompt_file_path = "sample_data_augmentation_system_prompt.txt"


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


def make_single_request(client: genai.Client, system_prompt: str, training_data: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.95,
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
    ai_client = create_client()

    for batch_index in range(NUMBER_OF_BATCHES):
        prompt = f"Generate {BATCH_SIZE} diverse training examples following the prompt rules."
        try:
            print(f"Processing batch [{batch_index}]...")
            response = make_single_request(ai_client, system_prompt, prompt)

        except Exception as e:
            print(f"Error processing batch {batch_index}: {e}")
        else:
            append_answers_to_file(response, OUTPUT_FILE_NAME)


if __name__ == "__main__":
    main()
