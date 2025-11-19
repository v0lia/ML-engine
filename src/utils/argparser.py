# argparser.py

import argparse
from pathlib import Path

from src.utils.checkpoint import find_latest_checkpoint
from src.utils.logger import get_logger
from src.utils.defaults import default_checkpoints_path, default_models_path


def parse_args(default_config_path):
    parser = argparse.ArgumentParser(description="Neural network train/eval engine")
    parser.add_argument("mode", choices=["train", "eval", "evaluate"],
                        help="Run mode: train or eval")
    parser.add_argument("-m", "--model", default=None,
                        help='In "train" mode: name of a model from src/models/, e.g. CNN')
    parser.add_argument("-c", "-cp", "--checkpoint", default=None,
                        help='Name of a checkpoint from checkpoints/ or results/<run_name>/checkpoints')
    parser.add_argument("-l", "--auto-latest", action="store_true",
                        help='In "eval" mode: autimatically loads the latest checkpoint')
    
    parser.add_argument("-cfg", "--config", default=default_config_path,
                        help=f"Path to YAML config file; default: {default_config_path}")
    
    #parser.add_argument("-r", "--run_name", default=None, 
    #                    help='Name of the experiment run (will be placed in "results/")')
    #parser.add_argument("-d", "--device", type=str, choices=['cpu','cuda'], default=None, help="Device: cpu or cuda")

    args = parser.parse_args()
    return args

def copy_cmdline_to_run_dir(line, run_dir):
    run_dir = Path(run_dir).resolve()
    dst = run_dir / "config" / "cmdline.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(" ".join(line))
    return dst

def resolve_mode(mode_arg) -> str:
    return "train" if mode_arg.lower() == "train" else "eval"    # choices=["train", "eval", "evaluate"]

def resolve_path(arg, is_checkpoint:bool) -> Path | None:
    if arg is None:
        return None
    path = Path(arg)
    ext = '.pt' if is_checkpoint else '.py'
    if path.suffix != ext:
        path = path.with_suffix(ext)
    if path.parent == Path('.'):
        path = (default_checkpoints_path if is_checkpoint else default_models_path) / path
    return path.resolve()

def resolve_checkpoint(checkpoint_arg) -> Path | None:
    return resolve_path(arg=checkpoint_arg, is_checkpoint=True)
  
def resolve_model(model_arg) -> Path | None:
    return resolve_path(arg=model_arg, is_checkpoint=False)

def resolve_checkpoint_or_model_path(mode_arg, checkpoint_arg, model_arg, auto_latest) -> tuple[Path, bool]:
    logger = get_logger()
    mode = resolve_mode(mode_arg=mode_arg)
    checkpoint_path = resolve_checkpoint(checkpoint_arg)
    model_path = resolve_model(model_arg)
    
    if mode == "train":
        if checkpoint_path and model_path:
            error_text = '[ARGPARSER] In "train" mode only one of [--checkpoint, --model] is allowed, e.g.: "python main.py train --model CNN"'
            logger.error(error_text)
            raise ValueError(error_text)
        if checkpoint_path:
            return (checkpoint_path, True)
        elif model_path:
            return (model_path, False)
        else:
            error_text = '[ARGPARSER] In "train" one of [--checkpoint, --model] is required, e.g.: "python main.py train --model CNN"'
            logger.error(error_text)
            raise ValueError(error_text)

    elif mode == "eval":
        if model_path:
            error_text = '[ARGPARSER] In "eval" mode --model is forbidden; architecture comes from --checkpoint'
            logger.error(error_text)
            raise ValueError(error_text)
        elif auto_latest and checkpoint_path:
            error_text = '[ARGPARSER] In "eval" mode only one of [--checkpoint, --auto_latest] is allowed, e.g.: "python main.py eval --auto-latest"'
            logger.error(error_text)
            raise ValueError(error_text)
        elif checkpoint_path:
            return (checkpoint_path, True)
        elif auto_latest:
            latest = find_latest_checkpoint()
            if latest is None:
                error_text = "[ARGPARSER] Not found any .pt files but --auto-latest given."
                logger.error(error_text)
                raise FileNotFoundError(error_text)
            return (latest, True)
        else:
            error_text = '[ARGPARSER] In "eval" one of [--checkpoint, --model] is required, e.g.: "python main.py eval --auto-latest"'
            logger.error(error_text)
            raise ValueError(error_text)
            
    else:
        error_text = f'[ARGPARSER] Unknown mode: {mode_arg}'
        logger.error(error_text)
        raise ValueError(error_text)
    
'''        
def resolve_run_dir(run_name_arg):
    
    return run_dir

def resolve_device(device_arg):

    return device
'''