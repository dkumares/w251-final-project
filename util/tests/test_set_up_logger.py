import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.insert(0, ROOT_DIR)

from util.set_up_logger import get_logger

RUN_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
RUN_TIME = datetime.now().strftime(RUN_TIME_FORMAT)

logger = get_logger(os.path.splitext(os.path.basename(__file__))[0], write_logs_to_file=True, run_time=RUN_TIME)

def main():    
    logger.info('Testing info logging.')
    logger.warn('Testing warn logging.')

if __name__ == '__main__':
    main()