import time
# import logging


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

    # for e in range(1, 11, 2):
    #    for fa in range(1, 11, 2):
    #        for fl in range(1, 11, 2):
    #            for fr in range(1, 11, 2):
    #                for fc in range(5):
    #                     for c in range(5):
    #                        for mtc in range(5):
    #                            #print("self.ds.addSample([%s], [%s]" % (" ".join(str(x) for x in self.roundfloat([e*0.1, fa*0.1, fl*0.1, fr*0.1, fc, c, mtc])), " ".join(str(x) for x in self.roundfloat(self.gettraining2([e*0.1, fa*0.1, fl*0.1, fr*0.1, fc, c, mtc])))))
    #                            self.ds.addSample([e*0.1, fa*0.1, fl*0.1, fr*0.1, fc, c, mtc], self.gettraining2([e*0.1, fa*0.1, fl*0.1, fr*0.1, fc, c, mtc]))
    # self.saveDS()
