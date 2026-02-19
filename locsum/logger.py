import logging

# Get a logger for this script
logger = logging.getLogger(__name__)


def setup_logging(level=logging.DEBUG):
    """Configure logging for this module"""
    # print() is for user consumption, logging is for developer consumption
    #logger.handlers.clear()  # Remove any existing handlers from your logger
    if not logger.handlers:  # Prevent duplicate handlers
        # TODO: Optionaly make the call to basicConfig if I need to
        handler = logging.StreamHandler()  # pass sys.stdout?
        handler.setLevel(level)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False  # Don't bubble up to root


