from src.utils.text import TextUtility
from src.config import get_config

utils = TextUtility(
    config=get_config("config.yaml")
)

utils.export_table("table.csv")