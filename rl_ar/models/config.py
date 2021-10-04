from typing import TypedDict


class ModelConfig(TypedDict):
    target: int
    num_hiddens: int


DEFAULT_CONFIG: ModelConfig = {
    "target": 10,
    "num_hiddens": 32,
}
