import logging


def setup_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
