# !/usr/bin/python
import motko
import multiprocessing
import logging
from common import timing_function
import argparse


@timing_function
def motkotrainer(filename, size, hiddenlayeramount, trainingloops, trainingamount, trainUntilConvergence=False, test=False):
    # self.threadLock.acquire()
    if(trainUntilConvergence):
        filename = ("%sConverge_%s_%s" % (filename.split('_')[0], filename.split('_')[1], filename.split('_')[2]))
    # motkoinstance = motko.motkowrapper("seppo_1_1.pkl", size, hiddenlayer, false)
    motkoinstance = motko.motkowrapper(filename, size, hiddenlayeramount, False)
    motkoinstance.checkiftrainingsetexits(test)
    loopstrained = 0
    # self.threadLock.release()trainUntilConvergencetrainUntilConvergence
    if(test):
        for _ in range(trainingloops):
            loopstrained += 1
            motkoinstance.trainfromfileds(trainingamount, trainUntilConvergence, test)
            motkoinstance.saveNNwithname("%s_%d_%s" % (filename.split('_')[0], (loopstrained * trainingamount), filename.split('_')[2]))
    elif(trainUntilConvergence):
            for _ in range(trainingloops):
                loopstrained += 1
                motkoinstance.motkolive.setname("%s_%d_%s" % (filename.split('_')[0], (loopstrained * trainingamount), filename.split('_')[2]))
                # motkoinstance.motkolive.TrainerCreateTrainingset()
                motkoinstance.trainfromfileds(trainingamount, trainUntilConvergence)
                # motkoinstance.motkolive.trainerTrainUntilConvergence()
                # motkoinstance.motkolive.saveDS()
                motkoinstance.saveNNwithname("%s_%d_%s" % (filename.split('_')[0], (loopstrained * trainingamount), filename.split('_')[2]))
    else:
        for _ in range(trainingloops):
            loopstrained += trainingamount
            print (loopstrained, trainingamount)
            motkoinstance.motkolive.setname("%s_%d_%s" % (filename.split('_')[0], (loopstrained * trainingamount), filename.split('_')[2]))
            # motkoinstance.motkolive.TrainerCreateTrainingset()
            motkoinstance.trainfromfileds(trainingamount, trainUntilConvergence)
            motkoinstance.saveNNwithname("%s_%d_%s" % (filename.split('_')[0], (loopstrained * trainingamount), filename.split('_')[2]))
            motkoinstance.saveNNwithname(filename)
    del motkoinstance


def trainmotkos(name, size, hiddenlayeramount, trainUntilConvergence=False, amount=10, trainingloops=100, trainingamount=100, test=False):
    trained = 0
    trainers = []
    maxtrainers = 3
    childthreads = 0

    # need to create ds first if missing
    motkoinstance = motko.motkowrapper("nakki", size, hiddenlayeramount)
    motkoinstance.checkiftrainingsetexits(test)
    del motkoinstance

    if(trainUntilConvergence):
        print ("Training %d %ss with trainUntilConvergence loops %d" % (amount, name, trainingloops))
    else:
        print ("Training %d %ss with training loops %d x training per loop %d" % (amount, name, trainingloops, trainingamount))
    for i in range(1, amount + 1):
        p = multiprocessing.Process(target=motkotrainer, args=("%s_%d_%d.pkl" % (name, trainingloops, i), [1024, 768], hiddenlayeramount, trainingloops, trainingamount, trainUntilConvergence, test))
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
    logging.info("motkotrainer start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="run testrun", action="store_true")
    parser.add_argument("--trainUntilConvergence", help="train untill converse, will take long time", action="store_true")
    parser.add_argument("--amount", help="how many motkos will be trained", type=int)
    parser.add_argument("--trainingloops", help="How many times Train or TrainUntilConvergence for one motko", type=int)
    parser.add_argument("--trainingamount", help="How many times motko is trained", type=int)
    parser.add_argument("--hiddenlayeramount", help="How many hiddenlayers, default 4", default=10, type=int)
    args = parser.parse_args()
    logging.info(args)
    if(args.test):
            trainmotkos("seppo", [1024, 768], hiddenlayeramount=args.hiddenlayeramount, trainUntilConvergence=False, amount=3, trainingloops=3, trainingamount=1, test=args.test)
    elif(args.amount is not None and args.amount is not None and args.trainingamount is not None):
        trainmotkos("seppo", [1024, 768], hiddenlayeramount=args.hiddenlayeramount, trainUntilConvergence=args.trainUntilConvergence, amount=args.amount, trainingloops=args.amount, trainingamount=args.trainingamount)
    else:
        parser.print_help()
