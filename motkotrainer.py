# !/usr/bin/python
import motko
import multiprocessing


def motkotrainer(filename, size, hiddenlayer, Trainingloops, trainUntilConvergence=False):
    # self.threadLock.acquire()
    print ("Start training", filename, size, hiddenlayer)
    motkoinstance = motko.motko(filename, size, hiddenlayer)
    # self.threadLock.release()trainUntilConvergencetrainUntilConvergence
    if(trainUntilConvergence):
        motkoinstance.train()
        motkoinstance.saveNN()
    else:
        for i in range(1, Trainingloops, 1):
            motkoinstance.trainloopamount(Trainingloops=i, printvalues=True)  # (10000*(hiddenneuron*hiddenlayer)))
            motkoinstance.saveNNwithname("%s_%d_%s" % (filename.split('_')[0], i, filename.split('_')[2]))


def trainmotkos(name, size, hiddenlayer, trainUntilConvergence=False, amount=10, Trainingloops=100):
    trained = 0
    trainers = []
    maxtrainers = 3
    childthreads = 0
    print ("Training %d %ss with training loops %d" % (amount, name, Trainingloops))
    for i in range(1, amount):
        p = multiprocessing.Process(target=motkotrainer, args=("%s_%d_%d.pkl" % (name, Trainingloops, i), [1024, 768], hiddenlayer, Trainingloops))
        trainers.append(p)
        p.start()
        childthreads += 1
        trained += 1
        if(childthreads == maxtrainers):
            for t in trainers:
                t.join()
                childthreads = 0
                if (trained > amount):
                    break
    for t in trainers:
        t.join()
    print (amount, trained)


if __name__ == "__main__":
    trainmotkos("seppo", [1024, 768], 5, trainUntilConvergence=False, amount=2, Trainingloops=100)
