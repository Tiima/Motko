# !/usr/bin/python

import os
import sys
import time
import random
import datetime
import pickle
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
import logging
from common import timing_function

"""
Motko "tuo oman tiensa kulkia"

"""


class motkowrapper:

    @timing_function
    def __init__(self, filename, eartsize, num_hiddeLayers, loadfromfile=False, test=False):
        logging.basicConfig(filename="motkowrapper.log", format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
        logging.info("motkowrapper start")
        self.dontdelete = True
        self.cwd = os.getcwd()
        self.filename = filename

        if(loadfromfile):
            self.motkolive = pickle.load(open(os.path.join(self.cwd, 'brains', self.filename), "rb"))
            if(self.dontdelete is False):
                os.remove(os.path.join(self.cwd, 'brains', self.filename))
        else:
            self.motkolive = motko(self.filename, eartsize, num_hiddeLayers, test)

    @timing_function
    def trainfromfileds(self, loops, trainUntilConvergence=False):
        self.motkolive.trainfromfileds(SupervisedDataSet.loadFromFile("basic_trainingset.ds"), loops, trainUntilConvergence)

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
        # TODO Randomize Hiden clcasses
        self.ds = SupervisedDataSet(input_amount, output_amount)
#        for i in range(16):
#            trainingsetup = format(i, '04b')
#            trainingresult = self.gettraining([int(trainingsetup[0]), int(trainingsetup[1]), int(trainingsetup[2]), int(trainingsetup[3])])
#            self.ds.addSample((int(trainingsetup[0]), int(trainingsetup[1]), int(trainingsetup[2]), int(trainingsetup[3])),
#                              (trainingresult[0], trainingresult[1], trainingresult[2], trainingresult[3]))

        self.nn = buildNetwork(input_amount, hidden_layers, output_amount, bias=True, hiddenclass=TanhLayer)  # todo randomize all layercalsses
        self.trainer = BackpropTrainer(self.nn, self.ds)

    @timing_function
    def TrainerCreateTrainingset(self):
        self.printlog("starting to create trainignset")
        sys.stdout.flush()
        for e in range(1, 11):
            for fa in range(1, 11, 2):
                for fl in range(1, 11, 2):
                    for fr in range(1, 11, 2):
                        for fc in range(5):
                            for c in range(5):
                                for mtc in range(5):
                                    # self.printlog("self.ds.addSample([%s], [%s]" % (" ".join(str(x) for x in self.roundfloat([e*0.1, fa*0.1, fl*0.1, fr*0.1, fc, c, mtc])), " ".join(str(x) for x in self.roundfloat(self.gettraining2([e*0.1, fa*0.1, fl*0.1, fr*0.1, fc, c, mtc])))))
                                    self.ds.addSample([e * 0.1, fa * 0.1, fl * 0.1, fr * 0.1, fc, c, mtc], self.gettraining2([e * 0.1, fa * 0.1, fl * 0.1, fr * 0.1, fc, c, mtc]))
        self.saveDS("basic_trainingset.ds")
        self.printlog("Create trainignset done")

    @timing_function
    def trainerTrainUntilConvergence(self):
        for i in range(1):
            self.printlog("before", self.trainer.train())
            sys.stdout.flush()
            self.trainer.trainUntilConvergence(validationProportion=0.2)
        self.printlog("after", self.trainer.train())
        sys.stdout.flush()

    @timing_function
    def trainloopamount(self, Trainingloops=1, printvalues=True):
        for i in range(Trainingloops - 1):
            # self.trainer.train()
            self.trainer.train()  # , self.nn.params)

        self.printlog(self.trainer.train())
        sys.stdout.flush()

    @timing_function
    def trainfromfileds(self, fileds, loops=10, trainUntilConvergence=False):
        self.printlog("starting training {}".format(len(fileds)))
        sys.stdout.flush()
        filedstrainer = BackpropTrainer(self.nn, fileds)
        self.printlog("Loading trainer done")
        sys.stdout.flush()
        if(trainUntilConvergence):
            self.printlog("starting trainUntilConvergence {} loops".format(loops))
            for i in range(loops):
                self.printlog("loop {} before {}".format(i, filedstrainer.train()))
                sys.stdout.flush()
                filedstrainer.trainUntilConvergence(validationProportion=0.2)
                self.printlog("loop {} after {}".format(i, filedstrainer.train()))
                sys.stdout.flush()
        else:
            self.printlog("starting training {} loops".format(loops))
            for i in range(loops):
                self.printlog(filedstrainer.train())
                sys.stdout.flush()

    @timing_function
    def saveDS(self, DSFilename):
        self.ds.saveToFile(DSFilename)

    @timing_function
    def responce(self, liveinput):
        trainingresult = self.gettraining2(liveinput)
        self.ds.addSample(liveinput, trainingresult)
        # self.printlog("%s: %s: %s" % (" ".join(str(x) for x in self.roundfloat(liveinput)), " ".join(str(x) for x in self.roundfloat(trainingresult)), " ".join(str(x) for x in self.roundfloat(self.nn.activate(liveinput)))))
        if(self.trainsteps == self.aftermovestrain):
            self.trainer = BackpropTrainer(self.nn, self.ds)
            # self.trainer.trainEpochs(1)
            # self.currenterror = self.trainer.train()
            # self.printlog("trainUntilConvergence1: %s" % (self.currenterror))
            # for  _ in range(10):
            #    self.trainer.trainUntilConvergence()
            self.currenterror = self.trainer.train()
            # self.printlog("trainUntilConvergence2 %s" % (self.currenterror))
            self.printlog("%s: %s: %s" % (" ".join(str(x) for x in self.roundfloat(liveinput)), " ".join(str(x) for x in self.roundfloat(trainingresult)), " ".join(str(x) for x in self.roundfloat(self.nn.activate(liveinput)))))
            self.trainsteps = 0
            self.trainings += 1
        # if(len(self.ds) == self.aftermovestrain):
        #        self.ds.saveToFile("ds_save_%s_%d.ds" % (self.filename, len(self.ds)))
        #        self.printlog("saved ds_save_%s_%d.ds" % (self.filename, len(self.ds)))
        #        self.aftermovestrain += self.aftermovestrain
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
        self.aftermovestrain = 5000
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

    @timing_function
    def saveLog(self, filename, strinki, fileaut):
        target = open(filename, fileaut)
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
            self.eyeleftplace[1] = self.Y - (self.size + 10)
            self.eyerightplace[0] = self.X + self.size
            self.eyerightplace[1] = self.Y + self.size + self.size
            self.eyesightsizeleft = [self.size, 10]
            self.eyesightsizeright = [self.size, 10]
        elif(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X + self.size + self.size
            self.eyeleftplace[1] = self.Y
            self.eyerightplace[0] = self.X
            self.eyerightplace[1] = self.Y + (self.size + self.size)
            self.eyesightsizeleft = [10, self.size]
            self.eyesightsizeright = [self.size, 10]
        elif(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X + (self.size + self.size)
            self.eyeleftplace[1] = self.Y + self.size
            self.eyerightplace[0] = self.X - (self.size + 10)
            self.eyerightplace[1] = self.Y + self.size
            self.eyesightsizeleft = [10, self.size]
            self.eyesightsizeright = [10, self.size]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X - (self.size + 10)
            self.eyeleftplace[1] = self.Y
            self.eyerightplace[0] = self.X
            self.eyerightplace[1] = self.Y + (self.size + self.size)
            self.eyesightsizeleft = [10, self.size]
            self.eyesightsizeright = [self.size, 10]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] == 0):
            self.eyeleftplace[0] = self.X - self.size
            self.eyeleftplace[1] = self.Y - (self.size + 10)
            self.eyerightplace[0] = self.X - self.size
            self.eyerightplace[1] = self.Y + (self.size + self.size)
            self.eyesightsizeleft = [self.size, 10]
            self.eyesightsizeright = [self.size, 10]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X
            self.eyeleftplace[1] = self.Y - (self.size + 10)
            self.eyerightplace[0] = self.X - (self.size + 10)
            self.eyerightplace[1] = self.Y
            self.eyesightsizeleft = [self.size, 10]
            self.eyesightsizeright = [10, self.size]
        elif(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X - (self.size + 10)
            self.eyeleftplace[1] = self.Y - self.size
            self.eyerightplace[0] = self.X + (self.size + self.size)
            self.eyerightplace[1] = self.Y - self.size
            self.eyesightsizeleft = [10, self.size]
            self.eyesightsizeright = [10, self.size]
        elif(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X
            self.eyeleftplace[1] = self.Y - (self.size + 10)
            self.eyerightplace[0] = self.X + (self.size + self.size)
            self.eyerightplace[1] = self.Y
            self.eyesightsizeleft = [self.size, 10]
            self.eyesightsizeright = [10, self.size]

    @timing_function
    def reinit(self):
        self.printlog("%s reinit" % (datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))))
        # self.pybrain_init(hidden_layers=random.randint(1, 40))
        self.randmovevector()
        self.seteyes()
        # self.nn.randomize()
        self.energy = 1
        self.X = random.randint(0, self.eartsize[0])
        self.Y = random.randint(0, self.eartsize[1])
        self.shadow[:] = []
        self.movecount = 0
        self.movememory = []
        # self.train()
        self.startime = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        self.printlog("%s reinit done" % (datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))))

    @timing_function
    def live(self, dontPrintInfo=False, Learning=True):

            self.eatamount = 0
            # inputs are: energy 0, food avail 1, food left 2, food right 3, food color 4, color 5, meeting motko color 6
            # print ([self.energy, self.foodavail, self.foodInLeft, self.foodInRight, self.foodcolor, self.colornumber, self.meetinmotkocolor])
            # printlog(self.energy)
            neuraloutputs = self.responce([self.energy, self.foodavail, self.foodInLeft, self.foodInRight, self.foodcolor, self.colornumber, self.meetinmotkocolor])

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
            if (self.eat != 0):
                if(self.foodavail != 0):
                    self.energy = self.energy + self.eatamount
            self.energy = self.energy - self.consumption

            if(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] == 0):
                self.randmovevector()
            self.speed = self.move * 3
            self.X += self.directionvector[self.direction][0] * int(self.speed)
            self.Y += self.directionvector[self.direction][1] * int(self.speed)
            self.seteyes()
            self.shadow.append([self.X, self.Y])
            if len(self.shadow) >= self.shadowlength:
                del self.shadow[0]

            if(self.turnleft > 0.001 or self.turnright > 0.001):  # othervice we would turn all the time , change in nessesary
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
                    self.seteyes()
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
                self.size = 2
            elif(self.energy > 2):
                self.size = 2
            else:
                self.size = int(self.energy * 6)
                if(self.size < 2):
                    self.size = 2

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
            roundedlist.append(round(rounuppilist[i], 4))
        return roundedlist

    @timing_function
    def getliveinfo(self):
        # time2 = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        # diff = time2 - self.startime
        return [round(self.energy, 4), self.filename.split('.')[0], self.movecount, self.trainings]
        # return [round(self.energy, 4), round(self.speed, 4), self.filename, diff.total_seconds(), self.movecount]

    @timing_function
    def areyouallive(self):
        time2 = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        diff = time2 - self.startime
        if(self.test):
            if(self.energy < -5.00 or self.energy > 5.00):
                return (["dood", self.energy, self.move, self.trainings])
            elif(diff.total_seconds() > 30):
                # self.saveLog(self.filename, self.nn.inspectTofile(), 'a+')
                return (["viable NN"])
            if(self.move == "exception dood"):
                return (["dood", self.move])
            return ["ok"]
        else:
            if(self.energy < -5.00 or self.energy > 5.00):
                self.energy = 0
                if (self.energy > 5.00):
                    # print ("Not viable motko, randomize %s" % (self.getliveinfo()))
                    # self.reinit()
                    return ["ok"]  # return ["dood"]
                else:
                    # print ("Not viable motko, randomize %s" % (self.getliveinfo()))
                    # self.reinit()
                    return ["ok"]  # return ["dood"]
            # elif(diff.total_seconds() > 900):
                # self.saveViableNN()
                # self.saveLog(self.filename, self.nn.inspectTofile(), 'a+')
                # return (["viable NN"])
            if(self.move == "exception dood"):
                return (["dood", self.move])
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
    def eatcalc(self, inputs):
        # EAT energy 0, food avail 1, food left 2, food right 3, food color 4, color 5,
        outputs = [0, 0, 0.1, 0, 0]  # default response
        if(inputs[0] < 0.75):  # hungry
            if(inputs[1] < inputs[2] or (inputs[4] == inputs[5] and inputs[4] != 4)):  # food in left more than front or food color is different than in your color meaning eatable
                # self.printlog("food in left more than front or food color is different than in your color meaning eatable")
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 0  # move
                outputs[3] = 1  # turn left
                outputs[4] = 0  # do not turn right
            elif(inputs[1] < inputs[3] or (inputs[4] == inputs[5] and inputs[4] != 4)):  # food in right more than front
                # self.printlog(" # food in right more than front")
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 0  # move
                outputs[3] = 0  # do not turn left
                outputs[4] = 1  # turn right
            elif(inputs[4] == inputs[5] and inputs[1] != 0):  # food is same color dont eat it will kill you
                # self.printlog("# food is same color dont eat it will kill you")
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 0  # move
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
        return outputs

    @timing_function
    def contactcalc(self, inputs):
        # if same color as you then sex or flee if different color flee or kill. by killing you get enegry what other has
        outputs = [0, 0, 0]  # default response
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
        return outputs

    @timing_function
    def gettraining2(self, inputs):
        # inputs are: energy 0, food avail 1, food left 2, food right 3, food color 4, color 5, meeting motko color 6,
        # outputs are: eat 0, eat amount 1, move 2, turn left 3, turn tight 4, kill 5, flee 6, sex 7
        # divide trainign to smaller parts
        eatoutputs = self.eatcalc(inputs)  # eat, eat amount, move, turn left, turn tight
        contactouputs = self.contactcalc(inputs)
        # print (inputs, (eatoutputs+contactouputs))
        return eatoutputs + contactouputs
