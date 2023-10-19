import logging
import inspect
import sys

from kit.time import now


logger = None  # python logger
LOG_INDENT_BASE = len(inspect.stack())  # base for logging indentation
START = None
MSG = None

def setup_logger(logging_level=logging.INFO, log_file=None):
    """ sets up the python logging system

    :param logging_level: logging level
    """

    global logger

    if logger is None:
        # initialize the logger
        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging_level)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        if log_file is not None:
            logger.addHandler(logging.FileHandler(log_file))


def log_info(msg, start=False, stop=False):
    if start:
        global START, MSG
        jetzt, START = now(return_obj=True)
        MSG = msg
        msg = f"START: {MSG}"
    elif stop:
        jetzt, STOP = now(return_obj=True)
        total_seconds = (STOP - START).total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        msg = f"STOP: {MSG} ({int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}) {msg}"
    else:
        jetzt = now()
    indentation = f"{indent()}"

    if type(msg) == list:
        text = f"{jetzt}: {indentation} {msg[0]}"
        if len(msg) > 1:
            len_jetzt = len(jetzt)
            len_indent = len(indentation)
            for nachricht in msg[1:]:
                text += f"\n{' '*len_jetzt}  {' ' * len_indent} {nachricht}"
    else:
        text = f"{jetzt}: {indentation} {msg}"

    if logger is not None:
        logger.info(text)
    else:
        print(text)


def log_warning(msg):
    text = f"{now()}: {indent()} WARNING: {msg}"
    if logger is not None:
        logger.warning(text)
    else:
        print(text)


def log_error(msg):
    text = f"{now()}: {indent()} ERROR: {msg}"
    if logger is not None:
        logger.error(text)
    else:
        print(text)


def log_caller(obj=None):
    """ logs the name of the calling function

    :param obj: object holding the calling function
    """

    c = caller(obj, indent=False)
    log_info(c)


def indent():
    """ returns the indentation for logging purposes

    :return: indentation as blank spaces string
    """

    return f"{' '*(len(inspect.stack()) - LOG_INDENT_BASE)}"


def caller(obj=None, indent=True):
    """ returns the name of the calling function for logging purposes

    :param obj: object holding the calling function
    :return: name of the calling function in log format
    """
    
    c = f"{obj.__class__.__name__}.{inspect.stack()[1][3]}" \
        if obj is not None \
        else f"{inspect.stack()[1][3]}"

    if indent:
        idt = indent()
        c = f"{idt} {c}"

    return c
