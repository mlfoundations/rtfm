def parse_unipredict_dataset_name(name: str) -> str:
    if name == "team-ai-spam-text-message-classification":
        return "team-ai/spam-text-message-classification"
    else:
        return name.replace("-", "/", 1)
