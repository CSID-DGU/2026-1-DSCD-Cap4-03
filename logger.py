import logging
import os
from datetime import datetime
import pytz
os.environ["TZ"] = "Asia/Seoul"

class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        kst = pytz.timezone("Asia/Seoul")

        dt = datetime.utcfromtimestamp(record.created).replace(tzinfo=pytz.utc)
        dt = dt.astimezone(kst)

        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        now = datetime.now().strftime("%y%m%d%H")
        log_path = os.path.join(save_dir, f"{now}.log")

        self.logger = logging.getLogger(f"train_logger_{now}")
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = KSTFormatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.propagate = False

    def info(self, msg):
        self.logger.info(msg)
