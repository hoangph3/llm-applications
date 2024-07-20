import redis


def remove_inst_tags(input_string: str):
    # Remote prompt instruct
    while '[INST]' in input_string:
        start_index = input_string.find('[INST]')
        end_index = input_string.rfind('[/INST]') + len('[/INST]')
        input_string = input_string[:start_index] + input_string[end_index:]
    return input_string


class RedisBackend:
    def __init__(self, uri="redis://localhost:6379/0"):
        self.uri = uri
        self.client = None
        self.connect()

    def connect(self):
        self.client = redis.Redis.from_url(self.uri)

    def put(self, key, value):
        if value is not None:
            value = str(value).encode()
        return self.client.rpush(key, value)

    def get(self, key, timeout=0):
        _, value = self.client.blpop(key, timeout)
        if value is not None:
            return value.decode()

    def delete(self, key):
        self.client.delete(key)

    def count(self, name):
        if '*' in name:
            return len(self.client.keys(name))
        return self.client.llen(name)