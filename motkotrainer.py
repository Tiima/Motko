# !/usr/bin/python
import motko
import multiprocessing


def motkotrainer(filename, size, hiddenlayer, Trainingloops, trainingamount, trainUntilConvergence=False):
    # self.threadLock.acquire()
    print ("Start training", filename, size, hiddenlayer)
    motkoinstance = motko.motko(filename, size, hiddenlayer)
    loopstrained = 0
    # self.threadLock.release()trainUntilConvergencetrainUntilConvergence
    if(trainUntilConvergence):
        motkoinstance.train()
        motkoinstance.saveNN()
    else:
        for _ in range(Trainingloops):
            loopstrained += trainingamount
            # print (loopstrained , trainingamount)
            motkoinstance.trainloopamount(Trainingloops=trainingamount, printvalues=True)  # (10000*(hiddenneuron*hiddenlayer)))
            motkoinstance.saveNNwithname("%s_%d_%s" % (filename.split('_')[0], loopstrained, filename.split('_')[2]))


def trainmotkos(name, size, hiddenlayer, trainUntilConvergence=False, amount=10, Trainingloops=100, trainingamount=100):
    trained = 0
    trainers = []
    maxtrainers = 3
    childthreads = 0
    print ("Training %d %ss with training loops %d x training per loop %d" % (amount, name, Trainingloops, trainingamount))
    for i in range(1, amount + 1):
        p = multiprocessing.Process(target=motkotrainer, args=("%s_%d_%d.pkl" % (name, Trainingloops, i), [1024, 768], hiddenlayer, Trainingloops, trainingamount))
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
    trainmotkos("seppo", [1024, 768], 5, trainUntilConvergence=False, amount=20, Trainingloops=10, trainingamount=10000)
