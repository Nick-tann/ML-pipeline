import sys
import logging

logger = logging.getLogger(__name__)
#Setup logger
def set_logger() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(asctime)s : %(message)s',datefmt = "%Y-%m-%dT%H:%M:%S%z")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    return