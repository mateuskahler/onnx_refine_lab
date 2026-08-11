
## Summary

This folder contains code to build data for fine-tunning a simple classifier.

The core idea is to use a LLM to correctly label the training dataset and use it to refine a smaller model.

The published repository uses sample/fake data at `sample_training_data.jsonl` and `sample_system_prompt.txt`. See those files to understand how to supply real data.

- Fill `training_data.jsonl` with input data,
- Fill `system_prompt.txt` with the instruction set,
- Run `reference_fetcher_Gemini/Groq/etc.py` to use the desired LLM to ground reference your data,
- Run `merge_reference_with_data.py` to merge training data with reference answer.

## Notes

### Dependencies
`python_requirements.txt` : you can filter out backends that you don't plan to use (eg.: remove Groq if using only Gemini)

### Gemini Response Format
(as observed at the time)
```py
sdk_http_response=HttpResponse(
  headers=<dict len=12>
) candidates=[Candidate(
  content=Content(
    parts=[
      Part(
        text='The model answer is here',
        thought_signature=b"\x01\x01\x01\x01\x01x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01"
      ),
    ],
    role='model'
  ),
  finish_reason=<FinishReason.STOP: 'STOP'>,
  index=0
)] create_time=None model_version='gemini-3.5-flash-lite' prompt_feedback=None response_id='rr_id_rrrrr_id' usage_metadata=GenerateContentResponseUsageMetadata(
  candidates_token_count=7,
  prompt_token_count=21,
  prompt_tokens_details=[
    ModalityTokenCount(
      modality=<MediaModality.TEXT: 'TEXT'>,
      token_count=21
    ),
  ],
  total_token_count=28
) model_status=None automatic_function_calling_history=[] parsed=None
```

### Groq Response Format
(as observed at the time)

```json
{
  "id": "chatcmpl-5c023456-1234-1234-1234-1234567890",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "here is the answer to the question: 02",
        "role": "assistant",
        "annotations": null,
        "executed_tools": null,
        "function_call": null,
        "reasoning": null,
        "tool_calls": null
      }
    }
  ],
  "created": 0000000000000,
  "model": "llama-3.1-8b-instant",
  "object": "chat.completion",
  "mcp_list_tools": null,
  "service_tier": "on_demand",
  "system_fingerprint": "fp_03030303",
  "usage": {
    "completion_tokens": 555,
    "prompt_tokens": 666,
    "total_tokens": 777,
    "completion_time": 0.056452476,
    "completion_tokens_details": null,
    "prompt_time": 0.063124425,
    "prompt_tokens_details": null,
    "queue_time": 0.332324861,
    "total_time": 0.120571101
  },
  "usage_breakdown": null,
  "x_groq": {
    "id": "req_01kzqav17me4q80nk9g7kgfr90",
    "debug": null,
    "seed": 42,
    "usage": null
  }
}
```