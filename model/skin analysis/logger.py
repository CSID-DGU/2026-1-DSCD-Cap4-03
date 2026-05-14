import logging
import os
from datetime import datetime

import pytz


class KSTFormatter(logging.Formatter):
    """Logging formatter that uses KST (UTC+9) timestamps."""

    def formatTime(self, record, datefmt=None):
        kst = pytz.timezone("Asia/Seoul")
        dt  = datetime.utcfromtimestamp(record.created).replace(tzinfo=pytz.utc)
        dt  = dt.astimezone(kst)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, save_dir: str, run_id: str = None):
        os.makedirs(save_dir, exist_ok=True)
        tag      = run_id or datetime.now().strftime("%y%m%d%H")
        log_path = os.path.join(save_dir, f"{tag}.log")

        self.logger = logging.getLogger(f"skin_logger_{tag}")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = KSTFormatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        for handler in [logging.FileHandler(log_path), logging.StreamHandler()]:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def info(self, msg: str):
        self.logger.info(msg)
