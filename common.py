import time
import logging


def timing_function(func):
    """
    Outputs the time a function takes
    to execute.
    """
    def wrapper(*args, **kwargs):
        t1 = time.time()
        response = func(*args, **kwargs)
        t2 = time.time()
        logging.info("Time it took to run the function: " + str((t2 - t1)))
        # print(" {}: {}".format(func.__name__ ,str((t2 - t1))))
        return response
    return wrapper
