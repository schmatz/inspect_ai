import json

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser


def prepare_dataset() -> MemoryDataset:
    with open("insider-trading/prompts/default.json") as f:
        default_prompt = json.load(f)

    message_mapping = {
        "system": ChatMessageSystem,
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
    }

    messages = [
        message_mapping[message["role"]](content=message["content"])
        for message in default_prompt["messages"]
    ]

    # We just have one sample in this dataset,
    # and will use epochs to iterate over it multiple times.
    samples = [
        Sample(
            input=messages,
            metadata={
                "misalignment_string": default_prompt["misalignment_string"],
                "deception_trigger": default_prompt["deception_trigger"],
                "deception_message_sep": default_prompt["deception_message_sep"],
                "doubling_down_trigger": default_prompt["doubling_down_trigger"],
                "doubling_down_sep": default_prompt["doubling_down_sep"],
            },
        )
    ]

    return MemoryDataset(
        samples=samples,
    )
