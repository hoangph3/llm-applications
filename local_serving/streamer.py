from transformers import TextStreamer


_STOP_SIGNAL = "EOL"
_EXCLUDE_SIGNAL = ['</s>']


class CustomStreamer(TextStreamer):
    def __init__(self, queue, _id, tokenizer, skip_prompt, **decode_kwargs) -> None:
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._queue = queue
        self._id = _id
        self.stop_signal = _STOP_SIGNAL
        self.timeout = 1
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._queue.put(self._id, text)
        if stream_end:
            self._queue.put(self._id, self.stop_signal)
