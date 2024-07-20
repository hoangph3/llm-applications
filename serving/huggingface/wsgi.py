from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from threading import Thread
import json
import time
import uuid

from utils import remove_inst_tags, RedisBackend
from load_model import load_model
from streamer import CustomStreamer, _STOP_SIGNAL, _EXCLUDE_SIGNAL
from schemas import ChatCompletionRequest, Message


app = FastAPI()
streamer_queue = RedisBackend(uri="redis://:difyai123456@localhost:6379/0")
model, tokenizer = load_model()


def generate(**kwargs):
    print(f'Generate inputs: {kwargs}')
    # Model inputs
    messages = kwargs.get('messages')
    request_id = kwargs.get('request_id')

    # Model params
    max_tokens = kwargs.get('max_tokens')
    frequency_penalty = kwargs.get('frequency_penalty')
    temperature = kwargs.get('temperature')
    top_p = kwargs.get('top_p')

    try:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        streamer = CustomStreamer(streamer_queue, request_id, tokenizer, True)
        outputs = model.generate(
            inputs=inputs, streamer=streamer, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=5,
            max_new_tokens=max_tokens,
            repetition_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p
        )
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = remove_inst_tags(decoded_output)
        print(answer)
    except Exception as e:
        error_text = f"Serving error: {e}"
        print(error_text)
        # Show error on chat & terminate
        streamer_queue.put(request_id, error_text)
        streamer_queue.put(request_id, _STOP_SIGNAL)


def start_generation(request_id, **kwargs):
    kwargs['request_id'] = request_id
    thread = Thread(
        target=generate,
        kwargs=kwargs
    ) 
    thread.start()


def response_generator(request_id, **kwargs):
    start_generation(request_id, **kwargs)
    i = 0

    while True:
        value = streamer_queue.get(request_id)
        if value is None or value in _EXCLUDE_SIGNAL:
            continue

        if value == _STOP_SIGNAL:
            streamer_queue.delete(request_id)
            break
        
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": kwargs.get('model'),
            "choices": [{"delta": {"content": value + " "}}],
        }
        i += 1
        yield f"data: {json.dumps(chunk)}\n\n"


def response_batch(request_id, **kwargs):
    start_generation(request_id, **kwargs)
    i = 0
    response_text = []

    while True:
        value = streamer_queue.get(request_id)
        if value is None or value in _EXCLUDE_SIGNAL:
            continue

        if value == _STOP_SIGNAL:
            streamer_queue.delete(request_id)
            break

        i += 1
        response_text.append(value)

    return i, "".join(response_text)


@app.post('/chat/completions')
def chat_completions(request: ChatCompletionRequest):

    # Check body
    if request.messages is None:
        raise HTTPException(400, "Messages is missing")

    # Limit num requests
    if streamer_queue.count('request_id:*') > 100:
        raise HTTPException(403, "Too many requests")

    request_id = f"request_id:{str(uuid.uuid4())}"
    kwargs = request.dict()

    if request.stream:
        # media_type = 'text/event-stream'
        media_type = 'application/x-ndjson'
        return StreamingResponse(response_generator(request_id, **kwargs), media_type=media_type)

    idx, resp_content = response_batch(request_id, **kwargs)
    return {
        "id": idx,
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=resp_content)}],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wsgi:app", host='0.0.0.0', port=8079)
