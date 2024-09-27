import torch


def load_model(model_path: str = "models/saved_models", model_name: str = "best.pt"):
    model = torch.load(model_path / model_name)
    return model

def save_model(model, model_path: str = "models/saved_models", model_name: str = "best.pt"):
    torch.save(model, model_path / model_name)