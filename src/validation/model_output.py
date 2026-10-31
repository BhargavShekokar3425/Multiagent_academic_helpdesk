"""Inspect final model content without exposing or accepting reasoning traces."""


def completion_output(response):
    choices = getattr(response, "choices", None) or []
    if not choices:
        return {"raw": None, "error": "Model returned no choices.",
                "response_metadata": {"failure": "no_choices"}}
    choice = choices[0]
    message = choice.message
    raw = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    usage = getattr(response, "usage", None)
    metadata = {
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_characters": len(raw) if isinstance(raw, str) else 0,
        "has_reasoning_content": bool(reasoning),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    if metadata["finish_reason"] == "length":
        return {"raw": None, "error": "Model reached its output limit; final JSON may be truncated.",
                "response_metadata": {**metadata, "failure": "output_limit"}}
    if not isinstance(raw, str) or not raw.strip():
        return {"raw": None, "error": "Model returned empty final content.",
                "response_metadata": {**metadata, "failure": "empty_final_content"}}
    return {"raw": raw, "error": None, "response_metadata": metadata}
