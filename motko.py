# !/usr/bin/python

import os
import sys
import time
import random
import datetime
import pickle
import logging
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import LinearLayer
from pybrain.structure import SigmoidLayer
from pybrain.structure import TanhLayer
# from pybrain.structure import GaussianLayer
from pybrain.structure import SoftmaxLayer
# from pybrain.structure import BiasUnit
from pybrain.structure import FullConnection
from pybrain.structure import FeedForwardNetwork
from common import timing_function
import shutil


class motkowrapper:
    """Motko "tuo oman tiensa kulkia"""

    @timing_function
    def __init__(self, filename, eartsize, num_hiddeLayers, loadfromfile=False, test=False):
        logging.basicConfig(filename="motkowrapper.log", format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
        logging.info("motkowrapper start")
        self.dontdelete = True
        self.cwd = os.getcwd()
        _, self.filename = os.path.split(filename)

        if(loadfromfile):
            self.motkolive = pickle.load(open(filename, "rb"))
            self.motkolive.reinit()
            if(self.dontdelete is False):
                os.remove(os.path.join(self.cwd, 'brains', self.filename))
        else:
            self.motkolive = motko(self.filename, eartsize, num_hiddeLayers, test)

    @timing_function
    def setNotViable(self):
        if(os.path.isdir(os.path.join(os.getcwd(), 'noVianble')) is not True):
            os.makedirs(os.path.join(os.getcwd(), 'noVianble'))
        shutil.move(os.path.join(os.getcwd(), 'brains', self.motkolive.filename), os.path.join(os.getcwd(), 'noVianble', '{}.movecnt{}.energy{}'.format(self.motkolive.filename, self.motkolive.movecount, self.motkolive.energy)))

    @timing_function
    def checkiftrainingsetexits(self, test=False):
        if(test):
            filename = "Basic_Test_TrainingSet.ds"
        else:
            filename = "Basic_TrainingSet.ds"
        if(os.path.isfile(os.path.join(self.cwd, filename)) is False):
            print("No {}, creating one".format(filename))
            self.motkolive.CreateTrainingset(test)

    @timing_function
    def trainfromfileds(self, loops, trainUntilConvergence=False, smallerTS=False):
        if(smallerTS):
            filename = "Basic_Test_TrainingSet.ds"
        else:
            filename = "Basic_TrainingSet.ds"
        if(os.path.isfile(os.path.join(self.cwd, filename)) is False):
            self.motkolive.CreateTrainingset(smallerTS)
        self.motkolive.trainfromfileds(SupervisedDataSet.loadFromFile(filename), loops, trainUntilConvergence)

    @timing_function
    def checkerror(self):
        if(self.motkolive.currenterror < 0.1):
            self.saveNNwithname("%s_%s_%s.%d.pkl" % (self.motkolive.filename.split('_')[0], self.motkolive.filename.split('_')[1], self.motkolive.filename.split('_')[2], self.motkolive.currenterror))

    @timing_function
    def saveNN(self):
        with open(os.path.join(self.cwd, 'brains', self.filename), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (self.filename))

    @timing_function
    def saveNNwithname(self, name):
        with open(os.path.join(self.cwd, 'brains', name), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (name))

    @timing_function
    def saveViableNN(self):
        with open(os.path.join(self.cwd, 'brains', (("%s.viable.pybrain_pkl") % (self.filename))), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (self.filename))

    @timing_function
    def saveEaterNN(self):
        with open(os.path.join(self.cwd, 'brains', (("%s.viable_eater.pybrain_pkl") % (self.filename))), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (self.filename))

    @timing_function
    def saveNotViableNN(self):
        with open(os.path.join(self.cwd, 'brains', (("%s.pkl_noviable") % (self.filename))), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (self.filename))


class motko:

    @timing_function
    def pybrain_init(self, input_amount=7, output_amount=8, hidden_layers=6):
        # TODO Randomize Hiden clcasses, ongoing...
        # because threading
        random.jumpahead(1252157)
        self.hiddenLayerAmount = random.randint(1, hidden_layers * 2)
        self.hiddenLayerNeuronsAmount = []
        # layerlist = [LinearLayer,SigmoidLayer,TanhLayer, GaussianLayer, SoftmaxLayer]  # for future use
        self.ds = SupervisedDataSet(input_amount, output_amount)
        self.nn = FeedForwardNetwork()
        self.inLayer = LinearLayer(input_amount, "in")
        # self.bias = BiasUnit(name="bias")
        if(random.randint(0, 100) >= 50):
            self.outLayer = LinearLayer(output_amount, "out")  # could be lineare layer or softmax???
        else:
            self.outLayer = SoftmaxLayer(output_amount, "out")  # could be lineare layer or softmax???
        self.hiddenlayers = []
        self.connections = []
        self.nn.addInputModule(self.inLayer)
        self.nn.addOutputModule(self.outLayer)
        # self.nn.addModule(self.bias)
        # self.nn.addConnection(FullConnection(self.inLayer, self.bias))

        for i in range(self.hiddenLayerAmount):  # example math.random(hidden_layers, hidden_layers*10)???
            self.hiddenLayerNeuronsAmount.append(random.randint(1, hidden_layers * 2))
            if(random.randint(0, 100) >= 50):
                self.hiddenlayers.append(TanhLayer(self.hiddenLayerNeuronsAmount[i], "hidden{}".format(i)))  # tanh or  sigmoid ??? and how many neurons ? now it is hidden_layers amount
            else:
                self.hiddenlayers.append(SigmoidLayer(self.hiddenLayerNeuronsAmount[i], "hidden{}".format(i)))  # tanh or  sigmoid ??? and how many neurons ? now it is hidden_layers amount

            if(i == 0):
                self.connections.append(FullConnection(self.inLayer, self.hiddenlayers[i - 1], name="in_to_hid"))
            else:
                self.connections.append(FullConnection(self.hiddenlayers[i - 1], self.hiddenlayers[i], name="hid{}_to_hid{}".format(i - 1, i)))
            self.nn.addModule(self.hiddenlayers[i])

        self.connections.append(FullConnection(self.hiddenlayers[len(self.hiddenlayers) - 1], self.outLayer, name="hid_to_out"))

        for i in range(len(self.connections)):
            self.nn.addConnection(self.connections[i])

        self.nn.sortModules()
        self.printlog(self.getliveinfo2())
        # self.printlog("hiddenLayerAmount:{}".format(self.hiddenLayerAmount))

    @timing_function
    def CreateTrainingset(self, smallerTS=False):
        if(smallerTS):
            self.printlog("starting to create trainignset")
            sys.stdout.flush()
            # inputs are: energy 0, food avail 1, food left 2, food right 3, food color 4, color 5, meeting motko color 6,
            e = -0.30
            fa = -0.20
            fl = -0.20
            fr = -0.20
            for _ in range(5):
                e = e + 0.25
                fa = -0.20
                fl = -0.20
                fr = -0.20
                for _ in range(5):
                    fa = fa + 0.25
                    fl = -0.20
                    fr = -0.20
                    for _ in range(5):
                        fl = fl + 0.25
                        fr = -0.20
                        for _ in range(6):
                            fr = fr + 0.25
                            for fc in range(5):
                                for c in range(5):
                                    for mtc in range(5):
                                        # self.printlog("self.ds.addSample([%s], [%s]" % (" ".join(str(x) for x in self.roundfloat([e, fa, fl, fr, fc, c, mtc])), " ".join(str(x) for x in self.roundfloat(self.gettraining2([e, fa, fl, fr, fc, c, mtc], self.nn.activate([e, fa, fl, fr, fc, c, mtc]))))))
                                        self.ds.addSample([e, fa, fl, fr, fc, c, mtc], self.gettraining2([e, fa, fl, fr, fc, c, mtc], self.nn.activate([e, fa, fl, fr, fc, c, mtc])))
            self.saveDS("Basic_Test_TrainingSet.ds")
            self.printlog("Create trainignset done")
            self.printlog("starting to create trainignset")
            sys.stdout.flush()
        else:
            for e in range(1, 11):
                for fa in range(1, 11, 2):
                    for fl in range(1, 11, 2):
                        for fr in range(1, 11, 2):
                            for fc in range(5):
                                for c in range(5):
                                    for mtc in range(5):
                                        # self.printlog("self.ds.addSample([%s], [%s]" % (" ".join(str(x) for x in self.roundfloat([e*0.1, fa*0.1, fl*0.1, fr*0.1, fc, c, mtc])), " ".join(str(x) for x in self.roundfloat(self.gettraining2([e*0.1, fa*0.1, fl*0.1, fr*0.1, fc, c, mtc], self.nn.activate([e, fa, fl, fr, fc, c, mtc])))))
                                        self.ds.addSample([e * 0.1, fa * 0.1, fl * 0.1, fr * 0.1, fc, c, mtc], self.gettraining2([e * 0.1, fa * 0.1, fl * 0.1, fr * 0.1, fc, c, mtc], self.nn.activate([e, fa, fl, fr, fc, c, mtc])))
            self.saveDS("Basic_TrainingSet.ds")
            self.printlog("Create trainignset done")

    @timing_function
    def trainerTrainUntilConvergence(self):
        for i in range(1):
            self.printlog("before", self.trainer.train())
            sys.stdout.flush()
            self.trainer.trainUntilConvergence(validationProportion=0.2)
        self.currenterror = self.trainer.train()
        self.printlog("after", self.trainer.train())
        sys.stdout.flush()

    @timing_function
    def trainloopamount(self, Trainingloops=1, printvalues=True):
        for i in range(Trainingloops - 1):
            # self.trainer.train()
            self.trainer.train()  # , self.nn.params)

        self.printlog(self.trainer.train())
        self.currenterror = self.trainer.train()
        sys.stdout.flush()

    @timing_function
    def trainfromfileds(self, fileds, loops=10, trainUntilConvergence=False):
        self.printlog("Loading training set {} samples long".format(len(fileds)))
        sys.stdout.flush()
        filedstrainer = BackpropTrainer(self.nn, fileds, learningrate=0.1, momentum=0.1)  # small learning rate should it be bigger?
        self.printlog("Loading training set done")
        sys.stdout.flush()
        if(trainUntilConvergence):
            self.printlog("Starting trainUntilConvergence {} loops".format(loops))
            for i in range(1, loops + 1):
                self.printlog("Loop {}, before error:{}".format(i, filedstrainer.train()))
                sys.stdout.flush()
                filedstrainer.trainUntilConvergence(validationProportion=0.2)
                self.printlog("Loop {}, after error:{}".format(i, filedstrainer.train()))
                sys.stdout.flush()
        else:
            self.printlog("Starting training {} loops".format(loops))
            for i in range(1, loops + 1):
                self.printlog("Loop {}, error:{}".format(i, filedstrainer.train()))
                sys.stdout.flush()

    @timing_function
    def saveDS(self, DSFilename):
        self.ds.saveToFile(DSFilename)

    @timing_function
    def responce(self, liveinput):
        self.trainingresult = self.gettraining2(liveinput, self.nn.activate(liveinput))
        if(self.trainsteps > self.trytolivesteps):
            self.ds.addSample(liveinput, self.trainingresult)
            if(self.trainsteps - self. trytolivesteps == self.aftermovestrain):
                self.trainer = BackpropTrainer(self.nn, self.ds, learningrate=0.6, momentum=0.1)  # small learning rate should it be bigger?
                # self.trainer.trainEpochs(1)
                # self.currenterror = self.trainer.train()
                # self.printlog("trainUntilConvergence1: %s" % (self.currenterror))
                if(self.test):
                    self.currenterror = self.trainer.train()
                else:
                    for _ in range(50):
                        self.trainer.trainUntilConvergence()
                    self.currenterror = self.trainer.train()
                    self.printlog(self.getliveinfo())
                # self.printlog("curren error {}".format(self.currenterror))
                # self.printlog("%s: %s: %s" % (" ".join(str(x) for x in self.roundfloat(liveinput)), " ".join(str(x) for x in self.roundfloat(self.trainingresult)), " ".join(str(x) for x in self.roundfloat(self.nn.activate(liveinput)))))
                self.trainsteps = 0
                self.trainings += 1
                self.trytolivesteps = 0
        if(len(self.ds) == 5000):
            self.ds.clear()
        return self.nn.activate(liveinput)

    @timing_function
    def collision(self, collider, collidersize):
        if (int(self.X) < int(collider[0]) + int(collidersize[0]) and int(self.X) + int(self.size[0]) > int(collider[0]) and int(self.Y) < int(collider[1]) + int(collidersize[1]) and int(self.size[1]) + int(self.Y) > int(collider[1])):
            # print (collider, self.X, self.Y)
            return 1  # collision
        else:
            return 0

    @timing_function
    def __init__(self, filename, eartsize, num_hiddeLayers, test=False):
        logging.basicConfig(filename="motkot.log", format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
        logging.info("motkot start")
        self.cwd = os.getcwd()
        self.test = test
        self.filename = filename
        self.pybrain_init(hidden_layers=num_hiddeLayers)

        self.eartsize = eartsize
        self.X = random.randint(0, eartsize[0])
        self.Y = random.randint(0, eartsize[1])
        self.consumption = 0.001
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        colors = []
        colors.append(self.RED)
        colors.append(self.BLACK)
        colors.append(self.GREEN)
        colors.append(self.BLUE)

        # inputs are: energy 0, food avail 1, food left 2, food right 3, food color 4, color 5, meeting motko color 6,
        self.energy = 0.9
        self.foodavail = 0
        self.foodInLeft = 0
        self.foodInRight = 0
        self.foodcolor = 4  # no food
        self.colornumber = random.randrange(3)
        self.meetinmotkocolor = 4  # no motko

        # outputs are: eat 0, eat amount 1, move 2, turn left 4, turn tight 5, kill 6, flee 7, sex 8
        self.eat = 0
        self.eatamount = 0
        self.move = 0
        self.turnleft = 0
        self.turnright = 0
        self.kill = 0
        self.flee = 0
        self.sex = 0

        self.color = colors[self.colornumber]
        self.shadow = []
        self.shadowlength = 100
        self.startime = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        self.movecount = 0
        self.movememory = []
        self.trainsteps = 0
        self.aftermovestrain = 100
        self.trytolivesteps = 2000
        self.randomcount = random.randint(5, 50)

        self.size = (5 + int(self.energy * 6))
        self.direction = 0
        self.directionvector = [] * 8
        self.directionvector.append([1, 0])
        self.directionvector.append([1, 1])
        self.directionvector.append([0, 1])
        self.directionvector.append([-1, 1])
        self.directionvector.append([-1, 0])
        self.directionvector.append([-1, -1])
        self.directionvector.append([0, -1])
        self.directionvector.append([1, -1])

        self.eyeleftplace = [] * 2
        self.eyeleftplace.append(0)
        self.eyeleftplace.append(1)

        self.eyerightplace = [] * 2
        self.eyerightplace.append(0)
        self.eyerightplace.append(1)
        self.eyesightsizeleft = [self.size, (self.size + 10)]
        self.eyesightsizeright = [self.size, (self.size + 10)]
        self.seteyes()
        self.randmovevector()
        self.trainings = 0
        self.currenterror = 10
        self.trainingresult = []
        self.doodReason = ""

    @timing_function
    def saveLog(self, filename, strinki, fileaut):
        if(os.path.isdir(os.path.join(os.getcwd(), 'logs')) is not True):
            os.makedirs(os.path.join(os.getcwd(), 'logs'))
        target = open(os.path.join(os.getcwd(), 'logs', filename), fileaut)
        if (not isinstance(strinki, str)):
            for item in strinki:
                target.write("%s\n" % item)
        else:
            target.write(strinki)
        target.close()

    @timing_function
    def seteyes(self):
        if(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] == 0):
            self.eyeleftplace[0] = self.X + self.size
            self.eyeleftplace[1] = self.Y - (self.size + 15)
            self.eyerightplace[0] = self.X + self.size
            self.eyerightplace[1] = self.Y + self.size + self.size
            self.eyesightsizeleft = [self.size, 15]
            self.eyesightsizeright = [self.size, 15]
        elif(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X + self.size + self.size
            self.eyeleftplace[1] = self.Y
            self.eyerightplace[0] = self.X
            self.eyerightplace[1] = self.Y + (self.size + self.size)
            self.eyesightsizeleft = [15, self.size]
            self.eyesightsizeright = [self.size, 15]
        elif(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X + (self.size + self.size)
            self.eyeleftplace[1] = self.Y + self.size
            self.eyerightplace[0] = self.X - (self.size + 15)
            self.eyerightplace[1] = self.Y + self.size
            self.eyesightsizeleft = [15, self.size]
            self.eyesightsizeright = [15, self.size]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X - (self.size + 15)
            self.eyeleftplace[1] = self.Y
            self.eyerightplace[0] = self.X
            self.eyerightplace[1] = self.Y + (self.size + self.size)
            self.eyesightsizeleft = [15, self.size]
            self.eyesightsizeright = [self.size, 15]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] == 0):
            self.eyeleftplace[0] = self.X - self.size
            self.eyeleftplace[1] = self.Y - (self.size + 15)
            self.eyerightplace[0] = self.X - self.size
            self.eyerightplace[1] = self.Y + (self.size + self.size)
            self.eyesightsizeleft = [self.size, 15]
            self.eyesightsizeright = [self.size, 15]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X
            self.eyeleftplace[1] = self.Y - (self.size + 15)
            self.eyerightplace[0] = self.X - (self.size + 15)
            self.eyerightplace[1] = self.Y
            self.eyesightsizeleft = [self.size, 15]
            self.eyesightsizeright = [15, self.size]
        elif(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X - (self.size + 15)
            self.eyeleftplace[1] = self.Y - self.size
            self.eyerightplace[0] = self.X + (self.size + self.size)
            self.eyerightplace[1] = self.Y - self.size
            self.eyesightsizeleft = [15, self.size]
            self.eyesightsizeright = [15, self.size]
        elif(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X
            self.eyeleftplace[1] = self.Y - (self.size + 15)
            self.eyerightplace[0] = self.X + (self.size + self.size)
            self.eyerightplace[1] = self.Y
            self.eyesightsizeleft = [self.size, 15]
            self.eyesightsizeright = [15, self.size]

    @timing_function
    def reinit(self):
        self.printlog("{} reinit".format(datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))))
        self.X = random.randint(0, self.eartsize[0])
        self.Y = random.randint(0, self.eartsize[1])
        self.randmovevector()
        self.seteyes()
        self.energy = 1
        self.shadow[:] = []
        self.movecount = 0
        self.startime = datetime.datetime.now()

    @timing_function
    def live(self, dontPrintInfo=False, test=False):

            self.test = test
            self.eatamount = 0
            # inputs are: energy 0, food avail 1, food left 2, food right 3, food color 4, color 5, meeting motko color 6
            # print ([self.energy, self.foodavail, self.foodInLeft, self.foodInRight, self.foodcolor, self.colornumber, self.meetinmotkocolor])
            # printlog(self.energy)
            neuraloutputs = self.responce([self.energy, self.foodavail, self.foodInLeft, self.foodInRight, self.foodcolor, self.colornumber, self.meetinmotkocolor])
            logging.info("{}: {}".format([self.energy, self.foodavail, self.foodInLeft, self.foodInRight, self.foodcolor, self.colornumber, self.meetinmotkocolor], neuraloutputs))
            if(dontPrintInfo):  # todo change that you can see output names
                self.printlog("\n%s\n%s: %f: %f: %d\n%s: %f: %d" % ("eat\t\teata\tmove\ttleft\ttright\tkill\tflee\tsex", "\t".join(str(x) for x in self.roundfloat(self.trainingresult)), self.foodavail, self.energy, self.colornumber, "\t".join(str(x) for x in self.roundfloat(neuraloutputs)), self.currenterror, len(self.ds)))
                # self.printlog("\n%s: \n%s: %f: %d" % (" ".join(str(x) for x in self.roundfloat([self.energy, self.foodavail, self.foodInLeft, self.foodInRight, self.foodcolor, self.colornumber, self.meetinmotkocolor])), " ".join(str(x) for x in self.roundfloat(neuraloutputs)), self.currenterror, len(self.ds)))

            self.eat = neuraloutputs[0]
            self.eatamount = neuraloutputs[1]
            self.move = neuraloutputs[2]
            self.turnleft = neuraloutputs[3]
            self.turnright = neuraloutputs[4]
            self.kill = neuraloutputs[5]
            self.flee = neuraloutputs[6]
            self.sex = neuraloutputs[7]
            # outputs are: eat 0, eat amount 1, move 2, turn left 4, turn tight 5, kill 6, flee 7, sex 8
            # eating
            if (self.eat > 0):
                if(self.foodavail > 0):
                    self.energy = self.energy + self.eatamount
                # if(self.foodcolor == self.colornumber):  # eatinmg wrong food
                    # printlog("dood by eating wrong color {} vs {}".format(self.foodcolor, self.colornumber))
                    # self.doodReason = "exception dood"
                    # return 1

            self.energy = self.energy - self.consumption

            if(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] == 0):
                self.randmovevector()

            self.speed = self.move * 10

            self.X += self.directionvector[self.direction][0] * int(self.speed)
            self.Y += self.directionvector[self.direction][1] * int(self.speed)

            self.shadow.append([self.X, self.Y])
            if len(self.shadow) >= self.shadowlength:
                del self.shadow[0]

            if(self.turnleft > 0.1 or self.turnright > 0.1):  # othervice we would turn all the time, change in nessesary
                if(self.turnleft > self.turnright):  # move left
                    if (self.direction % 2 == 0):
                        if(self.direction == 0):
                            self.direction = 6
                        else:
                            self.direction -= 2
                    elif(self.direction == 0):
                        self.direction = 7
                    else:
                        self.direction -= 1
                else:  # move right
                    if (self.direction % 2 == 0):
                        if(self.direction == 6):
                            self.direction = 0
                        else:
                            self.direction += 2
                    elif(self.direction == 7):
                        self.direction = 0
                    else:
                        self.direction += 1

            self.seteyes()

            # motko size
            if(self.energy < 0):
                self.size = int(0.001 * 6)
            elif(self.energy > 1.3):
                self.size = int(1.3 * 6)
            else:
                self.size = int(self.energy * 6)

            if(self.X >= self.eartsize[0]):
                self.X = 0
                self.randmovevector()
                self.seteyes()
            elif(self.X <= 0):
                self.X = self.eartsize[0]
                self.randmovevector()
                self.seteyes()
            if(self.Y >= self.eartsize[1]):
                self.Y = 0
                self.randmovevector()
                self.seteyes()
            if(self.Y <= 0):
                self.Y = self.eartsize[1]
                self.randmovevector()
                self.seteyes()

            self.foodavail = 0
            self.foodInLeft = 0
            self.foodInRight = 0
            self.meetinmotkocolor = 4
            self.foodcolor = 4
            self.movecount += 1
            self.trainsteps += 1

        # print (self.roundfloat(trainingoutputs), self.roundfloat(neuraloutputs), self.roundfloat([self.energy, self.foodavail]))

    @timing_function
    def didoueat(self):
        # print ("didoueat", self.eatamount)
        return self.eatamount

    @timing_function
    def addfoodavail(self, addfood, foodcolor):
        # self.printlog("addfoodavail",self.foodavail, self.foodcolor)
        self.foodavail = addfood
        self.foodcolor = foodcolor

    @timing_function
    def foodleft(self, foodInLeft):
        # self.printlog(self.foodInLeft, foodInLeft)
        self.foodInLeft = foodInLeft

    @timing_function
    def foodright(self, foodInRight):
        self.foodInRight = foodInRight

    @timing_function
    def randmovevector(self):
        vectorok = 1
        temp = random.randint(0, 7)
        while(vectorok):
            if(temp == self.direction):
                temp = random.randint(0, 7)
            else:
                self.direction = temp
                vectorok = 0

    @timing_function
    def roundfloat(self, rounuppilist):
        roundedlist = []
        for i in range(len(rounuppilist)):
            roundedlist.append('{:.3f}'.format(rounuppilist[i]))
        return roundedlist

    @timing_function
    def getliveinfo(self):
        # time2 = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        # diff = time2 - self.startime
        return [round(self.energy, 4), self.filename.split('.')[0], self.movecount, self.trainings, self.currenterror]
        # return [round(self.energy, 4), round(self.speed, 4), self.filename, diff.total_seconds(), self.movecount]

    @timing_function
    def getliveinfo2(self):
        returndata = "{}\n".format(self.filename.split('.')[0])
        returndata += "inLayer:{}\n".format(self.nn["in"])  # self.inLayer
        for i in range(len(self.hiddenlayers)):
            returndata += "\thidden{}:{}, neurons {}\n".format(i, self.hiddenlayers[i], self.hiddenLayerNeuronsAmount[i])
        returndata += "outLayer:{}\n".format(self.nn["out"])  # self.outLayer
        return returndata

    @timing_function
    def getliveinfo3(self):
        returndata = "{}\n".format(self.filename.split('.')[0])
        returndata += "inLayer:{}\n{}\n".format(self.nn["in"], self.connections[0].params)  # self.inLayer
        for i in range(len(self.hiddenlayers)):
            returndata += "\thidden{}:{}, neurons {}\n{}\n".format(i, self.hiddenlayers[i], self.hiddenLayerNeuronsAmount[i], self.connections[i + 1].params)

        returndata += "outLayer:{}\n{}\n".format(self.nn["out"], self.connections[len(self.connections) - 1].params)  # self.outLayer
        returndata += "error:{}\n".format(self.currenterror)
        return returndata

    @timing_function
    def areyouallive(self):
        time2 = datetime.datetime.now()
        diff = time2 - self.startime
        if(self.test):
            if(self.energy < -5.00 or self.energy > 5.00):
                return (["dood", self.energy, self.move, self.trainings])
            if(diff.total_seconds() > 120):
                # self.saveLog(self.filename, self.nn.inspectTofile(), 'a+')
                return (["viable NN"])
            if(self.doodReason == "exception dood"):
                return (["dood"])
            return ["ok"]
        else:
            if(self.energy < -5.00 or self.energy > 5.00):
                self.energy = 2
                # return ["dood"]
            elif (self.trainings == 1000):
                return ["dood"]
            if(self.doodReason == "exception dood"):
                return (["dood"])
            return ["ok"]

    @timing_function
    def getname(self):
        return self.filename

    @timing_function
    def setname(self, name):
        self.filename = name

    @timing_function
    def getinputaproximation(input):
        if(input <= 0.0):
            return 0
        elif(0.0 < input <= 0.25):
            return 1
        elif(0.25 < input <= 0.50):
            return 2
        elif(0.50 < input <= 0.75):
            return 3
        elif(0.75 < input <= 1):
            return 4
        elif(input > 1):
            return 5

    @timing_function
    def leftorright(left, right):
        if(left > right):
            return 0
        elif(left <= right):
            return 1
        elif(left == 0.01 and right == 0.01):
            return 2

    @timing_function
    def printlog(self, message):
        print("%s %s:%s" % (str(datetime.datetime.now()), self.filename, message))

    @timing_function
    def eatcalc(self, inputs, neuronoutputs=None):
        # EAT energy 0, food avail 1, food left 2, food right 3, food color 4, color 5,
        outputs = [99, 99, 99, 99, 99]  # default response
        if(inputs[0] < 0.75):  # hungry
            if(inputs[1] < inputs[2] or (inputs[4] == inputs[5] and inputs[4] != 4)):  # food in left more than front or food color is different than in your color meaning eatable
                # self.printlog("food in left more than front or food color is different than in your color meaning eatable")
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 0.25  # move
                outputs[3] = 1  # turn left
                outputs[4] = 0  # do not turn right
            elif(inputs[1] < inputs[3] or (inputs[4] == inputs[5] and inputs[4] != 4)):  # food in right more than front
                # self.printlog(" # food in right more than front")
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 0.25  # move
                outputs[3] = 0  # do not turn left
                outputs[4] = 1  # turn right
            elif(inputs[4] == inputs[5] and inputs[1] != 0):  # food is same color dont eat it will kill you
                # self.printlog("# food is same color dont eat it will kill you")
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 0.25  # move
                outputs[3] = 1  # turn right we prefer left
                outputs[4] = 0  # do not turn right
            elif(inputs[4] != inputs[5] and inputs[1] != 0):  # food is different color than you, it is eatable
                # self.printlog("# food is different color than you, it is eatable")
                if((inputs[0] + inputs[1]) < 1.5):  # food is not too mutch
                    # self.printlog("# food is not too mutch")
                    outputs[0] = 1  # eat
                    outputs[1] = inputs[1]  # eat all
                    outputs[2] = 0.25  # move
                    outputs[3] = 0  # turn right
                    outputs[4] = 0  # do not turn right
                    return outputs
                else:
                    # self.printlog("else # eat")
                    outputs[0] = 1  # eat
                    if(inputs[0] >= 0):
                        outputs[1] = (inputs[0] + inputs[1]) - 1  # Eat litle less
                    else:
                        outputs[1] = inputs[1]  # eat all
                    outputs[2] = 0.25  # move
                    outputs[3] = 0  # turn right
                    outputs[4] = 0  # do not turn right
                    return outputs
        if(0.75 < inputs[0] <= 1):  # hungry)
            outputs[2] = 0.25
        if(0.5 < inputs[0] <= 0.75):  # hungry)
            outputs[2] = 0.50
        if(0.25 < inputs[0] <= 0.50):  # hungry)
            outputs[2] = 0.75
        if(0.0 < inputs[0] <= 0.25):  # hungry)
            outputs[2] = 0.75
        if(0.0 > inputs[0]):  # hungry)
            outputs[2] = 1.25

        if(inputs[0] >= 1):  # full do not eat
            outputs[0] = 0  # dont eat
            outputs[1] = 0  # dont eat anything
        if(outputs[3] != 1 and outputs[4] != 1):  # dont turn
            outputs[3] = 0
            outputs[4] = 0
        if(inputs[4] == inputs[5] or inputs[4] == 4):  # definately dont eat
            outputs[0] = 0  # dont eat
            outputs[1] = 0  # dont eat anything

        for i in range(len(outputs)):
            if(outputs[i] == 99 and neuronoutputs is not None):  # if no matter use what you got from ANN
                outputs[i] = neuronoutputs[i]
        return outputs

    @timing_function
    def contactcalc(self, inputs, neuronoutputs=None):
        # if same color as you then sex or flee if different color flee or kill. by killing you get enegry what other has
        outputs = [99, 99, 99]  # default response
        if(inputs[5] == inputs[6]):  # same so flee or sex, energyamount defines what
            if(inputs[0] > 0.50):  # enought food to sex
                outputs[2] = inputs[0]
            else:
                outputs[1] = 1 - inputs[0]  # less amount food more fleeing
        else:
            if(inputs[0] > 0.5 and inputs[6] != 4):  # enought food to flee
                outputs[1] = inputs[0]
            elif(inputs[6] != 4):  # hungry so fight
                outputs[0] = 1 - inputs[0]

        for i in range(len(outputs)):
            if(outputs[i] == 99 and neuronoutputs is not None):
                outputs[i] = neuronoutputs[i + 4]

        return outputs

    @timing_function
    def gettraining2(self, inputs, neuronoutputs=None):
        # inputs are: energy 0, food avail 1, food left 2, food right 3, food color 4, color 5, meeting motko color 6,
        # outputs are: eat 0, eat amount 1, move 2, turn left 3, turn tight 4, kill 5, flee 6, sex 7
        # divide trainign to smaller parts
        eatoutputs = self.eatcalc(inputs, neuronoutputs)  # eat 0, eat amount 1, move 2, turn left 3, turn tight 4
        contactouputs = self.contactcalc(inputs, neuronoutputs)  # kill 5, flee 6, sex 7
        # print (inputs, (eatoutputs+contactouputs))
        return eatoutputs + contactouputs
