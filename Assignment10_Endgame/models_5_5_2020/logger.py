import logging

from datetime import datetime

from colorlog import ColoredFormatter



def get_logger(path=None):
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(green)s %(filename)s - %(log_color)s%(process)d - %(green)s%(asctime)s - %(log_color)s%(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
        secondary_log_colors={},
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logstamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    log_file_name = f"{path}/endgame_{logstamp}.log"
    
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
