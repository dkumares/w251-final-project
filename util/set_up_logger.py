import logging
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, ROOT_DIR)

def get_console_logger(formatter, name):
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(console)

    return console


def get_logger(name, write_logs_to_file=True, run_time=None):
    log_format='%(asctime)s - %(name)s - %(levelname)s: [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format)

    if not write_logs_to_file:
        get_console_logger(formatter=formatter, name=name)
    else:
        # create with a file handler for storing logs
        if run_time is None:
            raise ValueError('Run time must be configured for writing logs to file.')
        logging.basicConfig(level=logging.INFO,
                            format=log_format,
                            filename=f'{name}_{run_time}.log',
                            filemode='w')
        get_console_logger(formatter=formatter, name=name)
    return logging.getLogger(name)