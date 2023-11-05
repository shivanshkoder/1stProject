import logging
import os
import sys
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs")

os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="a",
    format="[%(asctime)s] %(lineno)d- %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)