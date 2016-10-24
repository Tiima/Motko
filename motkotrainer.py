# !/usr/bin/python
import motko
import multiprocessing
import logging
from common import timing_function
import argparse
import os


@timing_function
def motkotrainer(filename, size, hiddenlayeramount, trainingloops, trainingamount, filename2, trainUntilConvergence=False, smallerTS=False):
    # self.threadLock.acquire()
    if(trainUntilConvergence):
        filename = ("%sConverge_%s_%s" % (filename.split('_')[0], filename.split('_')[1], filename.split('_')[2]))
    # motkoinstance = motko.motkowrapper("seppo_1_1.pkl", size, hiddenlayer, false)
    if(os.path.isfile(os.path.join(os.getcwd(), 'brains', filename)) is True):
        motkoinstance = motko.motkowrapper(os.path.join(os.getcwd(), 'brains', filename), size, hiddenlayeramount, True)
        filename = filename2
        loopstrained = int(filename.split('_')[1])
    else:
        motkoinstance = motko.motkowrapper(filename, size, hiddenlayeramount, False)
        loopstrained = 0
    # self.threadLock.release()trainUntilConvergencetrainUntilConvergence
    motkoinstance.motkolive.saveLog("{}.log".format(motkoinstance.motkolive.filename), "Start training\n", 'a+')
    motkoinstance.motkolive.saveLog("{}.log".format(motkoinstance.motkolive.filename), motkoinstance.motkolive.getliveinfo2(), 'r+')
    motkoinstance.motkolive.saveLog("{}.log".format(motkoinstance.motkolive.filename), motkoinstance.motkolive.getliveinfo3(), 'a+')
    if(trainUntilConvergence):
            for _ in range(trainingloops):
                loopstrained += trainingamount
                filenametemp = "%s_%d_%s" % (filename.split('_')[0], (loopstrained), filename.split('_')[2])
                motkoinstance.motkolive.setname(filenametemp)
                motkoinstance.trainfromfileds(trainingamount, trainUntilConvergence, smallerTS)
                motkoinstance.saveNNwithname(filenametemp)
    else:
        for _ in range(trainingloops):
                loopstrained += trainingamount
                filenametemp = "%s_%d_%s" % (filename.split('_')[0], (loopstrained), filename.split('_')[2])
                motkoinstance.motkolive.setname(filenametemp)
                motkoinstance.trainfromfileds(trainingamount, trainUntilConvergence, smallerTS)
                motkoinstance.saveNNwithname(filenametemp)
    motkoinstance.motkolive.saveLog("{}.log".format(motkoinstance.motkolive.filename), "Stop training\n", 'a+')
    motkoinstance.motkolive.saveLog("{}.log".format(motkoinstance.motkolive.filename), motkoinstance.motkolive.getliveinfo2(), 'a+')
    motkoinstance.motkolive.saveLog("{}.log".format(motkoinstance.motkolive.filename), motkoinstance.motkolive.getliveinfo3(), 'a+')
    del motkoinstance


def trainmotkos(filename, size, hiddenlayeramount, trainUntilConvergence=False, amount=10, trainingloops=100, trainingamount=100, smallerTS=False):
    trained = 0
    trainers = []
    maxtrainers = 3
    childthreads = 0

    # need to create ds first if missing
    motkoinstance = motko.motkowrapper("nakki", size, hiddenlayeramount)
    motkoinstance.checkiftrainingsetexits(smallerTS)
    del motkoinstance

    if(trainUntilConvergence):
        print ("Training %d %s with trainUntilConvergence loops %d" % (amount, filename, trainingloops))
    else:
        print ("Training %d %s with training loops %d x training per loop %d" % (amount, filename, trainingloops, trainingamount))
    if (os.path.isfile(os.path.join(os.getcwd(), "brains", filename)) is True):
        print("training only one motko")
        tempfilename = "%s_%s_%d.pkl" % (filename.split('_')[0], filename.split('_')[1], (int(filename.split('_')[2].split(".")[0])))
        filename2 = "%s_%s_%d.pkl" % (filename.split('_')[0], filename.split('_')[1], (i + int(filename.split('_')[2].split(".")[0])))
        p = multiprocessing.Process(target=motkotrainer, args=(tempfilename, size, hiddenlayeramount, trainingloops, trainingamount, filename2, trainUntilConvergence, smallerTS))
        i = amount + 1   # train same motko three times? waste total waste
        trainers.append(p)
        p.start()
    else:
        for i in range(1, amount + 1):
            p = multiprocessing.Process(target=motkotrainer, args=("%s_%d_%d.pkl" % (filename.split('_')[0], trainingloops, i), size, hiddenlayeramount, trainingloops, trainingamount, "", trainUntilConvergence, smallerTS))
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
    logging.debug("motkotrainer start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--smallerTS", help="Train with smaller test set", action="store_true")
    parser.add_argument("--trainUntilConvergence", help="train untill converse, will take long time", action="store_true")
    parser.add_argument("--amount", help="how many motkos will be trained", type=int)
    parser.add_argument("--trainingloops", help="How many times Train or TrainUntilConvergence for one motko", type=int)
    parser.add_argument("--trainingamount", help="How many times motko is trained", type=int)
    parser.add_argument("--hiddenlayeramount", help="How many hiddenlayers, default 4", default=10, type=int)
    parser.add_argument("--motko", help="already created Motko to be used, file name", action="store", dest="motko")
    args = parser.parse_args()
    logging.info(args)
    print(args)
    if(args.amount is not None and args.amount is not None and args.trainingamount is not None and args.motko is not None):
        trainmotkos(filename=args.motko, size=[1024, 768], hiddenlayeramount=args.hiddenlayeramount, trainUntilConvergence=args.trainUntilConvergence,
                    amount=args.amount, trainingloops=args.trainingloops, trainingamount=args.trainingamount, smallerTS=args.smallerTS)
    else:
        parser.print_help()
