import logging

'''
Loggers hierarchy is the following:

- tritondse.
    - 

'''


logger = logging.getLogger('tritondse')
logger.propagate = False  # Do not propagate logs by default
color_enabled = True
_loggers = {}

def get(name: str = "") -> logging.Logger:
    """
    Get a child logger from the tritondse one.
    If name is empty returns the root tritondse
    logger.

    :param name: logger name
    """
    log = logger.getChild(name) if name else logger

    if log.name not in _loggers:
        log.propagate = False  # first time it is retrieve disable propagation
        _loggers[log.name] = log

    return log


def enable(level: int = logging.INFO, name: str = "") -> None:
    """
    Enable tritondse logging to terminal output

    :param level: logging level
    :param name: name of the logger to enable (all by default)
    """
    log = get(name)
    log.propagate = True
    log.setLevel(level)

    # Enable root logger if needed
    if log.name != "tritondse":
        logger.propagate = True
    else:
        for sub_logger in _loggers.values():
            sub_logger.propagate = True


def enable_to_file(level: int, file: str, name: str = "") -> None:
    """
    Enable tritondse logging to a file

    :param level: logging level
    :param file: path to log file
    :param name: name of the logger to enable to a file
    """
    log = get(name)
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(threadName)s [%(levelname)s] %(message)s")

    handler = logging.FileHandler(file)
    handler.setFormatter(fmt)
    log.addHandler(handler)  # Add the handler to the logger