import logging
from config.settings import get_config

def setup_logging():
    config = get_config()
    logging.basicConfig(
        filename=config.LOG_PATH,
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)