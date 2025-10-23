from pathlib import Path
import importlib

def get_model_class(model: str):
    model = Path(model.strip()).stem
    try:
        module = importlib.import_module(f"src.models.{model}")
        return getattr(module, "NeuralNetwork")
    except ModuleNotFoundError:
        raise FileNotFoundError(f"Model file {model}.py not found in src/models/")
    except AttributeError:
        raise ValueError(f"Model class in {model} must be named NeuralNetwork.")
