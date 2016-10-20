# !/usr/bin/python
import motko
import multiprocessing
import sys
import logging
from common import timing_function


@timing_function
def motkotrainer(filename, size, hiddenlayer, Trainingloops, trainingamount, trainUntilConvergence=False):
    # self.threadLock.acquire()
    if(trainUntilConvergence):
        filename = ("%sConverge_%s_%s" % (filename.split('_')[0], filename.split('_')[1], filename.split('_')[2]))
    # motkoinstance = motko.motkowrapper("seppo_1_1.pkl", size, hiddenlayer, false)
    motkoinstance = motko.motkowrapper(filename, size, hiddenlayer, False)
    # print (motkoinstance)
    loopstrained = 0
    # self.threadLock.release()trainUntilConvergencetrainUntilConvergence
    if(trainUntilConvergence):
            for _ in range(Trainingloops):
                loopstrained += 1
                motkoinstance.motkolive.setname("%s_%d_%s" % (filename.split('_')[0], (loopstrained * 10), filename.split('_')[2]))
                motkoinstance.trainfromfileds(trainingamount, trainUntilConvergence)
                # motkoinstance.motkolive.trainerTrainUntilConvergence()
                # motkoinstance.motkolive.saveDS()
                motkoinstance.saveNNwithname("%s_%d_%s" % (filename.split('_')[0], (loopstrained * 10), filename.split('_')[2]))
    else:
        for _ in range(Trainingloops):
            loopstrained += trainingamount
            print (loopstrained, trainingamount)
            motkoinstance.motkolive.setname("%s_%d_%s" % (filename.split('_')[0], (loopstrained * 10), filename.split('_')[2]))
            # motkoinstance.trainfromfileds(1, trainUntilConvergence)
            motkoinstance.motkolive.trainloopamount(trainingamount)
            # motkoinstance.saveNNwithname("%s_%d_%s" % (filename.split('_')[0], loopstrained, filename.split('_')[2]))
            motkoinstance.saveNNwithname(filename)
    del motkoinstance


@timing_function
def trainmotkos(name, size, hiddenlayer, trainUntilConvergence=False, amount=10, Trainingloops=100, trainingamount=100):
    trained = 0
    trainers = []
    maxtrainers = 3
    childthreads = 0
    if(trainUntilConvergence):
        print ("Training %d %ss with trainUntilConvergence loops %d" % (amount, name, Trainingloops))
    else:
        print ("Training %d %ss with training loops %d x training per loop %d" % (amount, name, Trainingloops, trainingamount))
    for i in range(1, amount + 1):
        p = multiprocessing.Process(target=motkotrainer, args=("%s_%d_%d.pkl" % (name, Trainingloops, i), [1024, 768], hiddenlayer, Trainingloops, trainingamount, trainUntilConvergence))
        trainers.append(p)
        p.start()
        childthreads += 1
        trained += 1
        if(childthreads == maxtrainers):
            for t in trainers:
                t.join()
                childthreads = 0
    for t in trainers:
        t.join()
    print (amount, trained)


if __name__ == "__main__":
    logging.basicConfig(filename="motkotrainer.log", format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.error("motkotrainer start")
    if(len(sys.argv) == 2):
            if("test" in sys.argv[1]):
                trainmotkos("seppo", [1024, 768], 7, trainUntilConvergence=True, amount=3, Trainingloops=3, trainingamount=10000)
    else:
        trainmotkos("seppo", [1024, 768], 7, trainUntilConvergence=True, amount=10, Trainingloops=1, trainingamount=10)
