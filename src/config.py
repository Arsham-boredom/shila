from omegaconf import OmegaConf

def get_config(file_path) -> OmegaConf:
    return OmegaConf.load(file_path)