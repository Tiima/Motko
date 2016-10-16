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


"""
Motko "tuo oman tiensa kulkia"

"""


class motkowrapper:

    def __init__(self, filename, eartsize, num_hiddeLayers, loadfromfile=False, test=False):

        self.dontdelete = True
        self.cwd = os.getcwd()
        self.filename = filename

        if(loadfromfile):
            # print ("loading motko ",  self.filename)
            self.motkolive = pickle.load(open(os.path.join(self.cwd, 'brains', self.filename), "rb"))
            if(self.dontdelete is False):
                os.remove(os.path.join(self.cwd, 'brains', self.filename))
        else:
            self.motkolive = motko(self.filename, eartsize, num_hiddeLayers, test)

    def saveNN(self):
        with open(os.path.join(self.cwd, 'brains', self.filename), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (self.filename))

    def saveNNwithname(self, name):
        with open(os.path.join(self.cwd, 'brains', name), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (name))

    def saveViableNN(self):
        with open(os.path.join(self.cwd, 'brains', (("%s.viable.pybrain_pkl") % (self.filename))), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (self.filename))

    def saveEaterNN(self):
        with open(os.path.join(self.cwd, 'brains', (("%s.viable_eater.pybrain_pkl") % (self.filename))), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (self.filename))

    def saveNotViableNN(self):
        with open(os.path.join(self.cwd, 'brains', (("%s.pkl_noviable") % (self.filename))), 'wb') as output:
            pickle.dump(self.motkolive, output, pickle.HIGHEST_PROTOCOL)
        print ("%s saved" % (self.filename))


class motko:

    def gettraining(self, inputs):
        # print ("calculated_output", calculated_output)
        training_outputs = [] * 4
        # print ("gettraining, inputs", inputs)
        if(inputs == [1, 0, 0, 0]):  # full
            training_outputs.append(0)
            training_outputs.append(1)  # move
            training_outputs.append(0)
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [1, 1, 0, 0]):  # hungri foo avail
            training_outputs.append(1)  # eat
            training_outputs.append(0)
            training_outputs.append(0)
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [1, 1, 1, 0]):  # hungri foodavail food in left
            training_outputs.append(1)  # eat
            training_outputs.append(0)
            training_outputs.append(1)  # turn
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [1, 1, 1, 1]):  # hungri foodavail food in left and right
            training_outputs.append(1)  # eat
            training_outputs.append(0)
            training_outputs.append(1)  # prefer left
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [1, 1, 0, 1]):  # hungri foodavail food in right
            training_outputs.append(1)  # eat
            training_outputs.append(0)
            training_outputs.append(0)
            training_outputs.append(1)  # turn
            return training_outputs
        elif(inputs == [0, 1, 1, 1]):  # full foodavail food in left and right
            training_outputs.append(0)
            training_outputs.append(0)
            training_outputs.append(0)
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [0, 0, 1, 1]):  # full food in left and right
            training_outputs.append(0)
            training_outputs.append(0)  # should we move?
            training_outputs.append(1)  # prefer left
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [0, 0, 0, 1]):  # food in right
            training_outputs.append(0)
            training_outputs.append(0)
            training_outputs.append(0)
            training_outputs.append(1)  # turn
            return training_outputs
        elif(inputs == [1, 0, 1, 0]):  # hungri food in left
            training_outputs.append(0)
            training_outputs.append(1)  # move
            training_outputs.append(1)  # turn
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [1, 0, 0, 1]):  # hungri food in right
            training_outputs.append(0)
            training_outputs.append(1)  # move
            training_outputs.append(0)
            training_outputs.append(1)  # turn
            return training_outputs
        elif(inputs == [1, 0, 1, 1]):  # hungri food in left and right
            training_outputs.append(0)
            training_outputs.append(1)  # move
            training_outputs.append(1)  # we prefer left
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [0, 1, 0, 0]):  # full foodavail
            training_outputs.append(0)
            training_outputs.append(1)  # should we prefer stop and wait until hungri again?
            training_outputs.append(0)
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [0, 0, 1, 0]):  # full food in left
            training_outputs.append(0)
            training_outputs.append(1)  # move
            training_outputs.append(1)  # tur
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [0, 0, 0, 0]):  # full no foods
            training_outputs.append(0)
            training_outputs.append(1)  # move
            training_outputs.append(0)
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [0, 1, 0, 1]):  # foodavail food in right
            training_outputs.append(0)
            training_outputs.append(1)
            training_outputs.append(0)
            training_outputs.append(1)
            return training_outputs
        elif(inputs == [0, 1, 1, 0]):
            training_outputs.append(0)
            training_outputs.append(1)  # move
            training_outputs.append(1)  # we prefer left
            training_outputs.append(0)
            return training_outputs
        elif(inputs == [0, 0, 0, 0]):  # hungri foo avail
            training_outputs.append(0)  # eat
            training_outputs.append(1)
            training_outputs.append(0)
            training_outputs.append(0)
            return training_outputs
        else:
            print ("fail!!!", inputs)

    def pybrain_init(self, input_amount=4, output_amount=4, hidden_layers=4):
        # TODO Randomize Hiden clcasses
        self.ds = SupervisedDataSet(input_amount, output_amount)
        for i in range(16):
            trainingsetup = format(i, '04b')
            trainingresult = self.gettraining([int(trainingsetup[0]), int(trainingsetup[1]), int(trainingsetup[2]), int(trainingsetup[3])])
            self.ds.addSample((int(trainingsetup[0]), int(trainingsetup[1]), int(trainingsetup[2]), int(trainingsetup[3])),
                              (trainingresult[0], trainingresult[1], trainingresult[2], trainingresult[3]))

        self.nn = buildNetwork(4, 5, 4, bias=True, hiddenclass=TanhLayer)
        self.trainer = BackpropTrainer(self.nn, self.ds)

    def train(self):
        for i in range(1):
            # self.trainer.train()
            self.trainer.trainUntilConvergence(validationProportion=0.2)
        print(self.trainer.train())

    def trainloopamount(self, Trainingloops=1, printvalues=True):
        for i in range(Trainingloops - 1):
            # self.trainer.train()
            self.trainer.train()  # , self.nn.params)

        print(self.trainer.train())

    def responce(self, liveinput):
        try:
            trainingresult = self.gettraining([int(liveinput[0]), int(liveinput[1]), int(liveinput[2]), int(liveinput[3])])

            self.ds.addSample((int(liveinput[0]), int(liveinput[1]), int(liveinput[2]), int(liveinput[3])),
                              (trainingresult[0], trainingresult[1], trainingresult[2], trainingresult[3]))
            if(self.trainsteps == self.aftermovestrain + self.randomcount):
                self.trainer = BackpropTrainer(self.nn, self.ds)
                # self.trainer.train()
                self.trainer.trainEpochs(2)
                # self.trainer.trainUntilConvergence(validationProportion=0.2)
                self.trainsteps = 0
            if(len(self.ds) == 200):
                self.ds.clear()
            return self.nn.activate((liveinput[0], liveinput[1], liveinput[2], liveinput[3]))
        except:
            print ("Unexpected error:", sys.exc_info())

    def __init__(self, filename, eartsize, num_hiddeLayers, test=False):
        self.cwd = os.getcwd()
        self.test = test
        self.filename = filename
        self.pybrain_init(hidden_layers=num_hiddeLayers)
        self.foodamount = 0.9
        self.eartsize = eartsize
        self.X = random.randint(0, eartsize[0])
        self.Y = random.randint(0, eartsize[1])
        self.consumption = 0.01
        self.foodavail = 0
        self.shadow = []
        self.shadowlength = 100
        self.startime = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        self.move = "moved"
        self.eaten = "full"
        self.speed = int(1)

        # print (self.X, self.Y)
        self.movecount = 0
        self.movememory = []
        self.trainsteps = 0
        self.aftermovestrain = 100
        self.randomcount = random.randint(5, 50)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        colors = []
        colors.append(self.RED)
        colors.append(self.BLACK)
        colors.append(self.GREEN)
        colors.append(self.BLUE)
        self.color = colors[random.randint(0, 3)]
        self.size = (5 + int(self.foodamount * 6))
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
        self.foodInLeft = 0
        self.eyeleftplace = [] * 2
        self.eyeleftplace.append(0)
        self.eyeleftplace.append(1)
        self.foodInRight = 0
        self.eyerightplace = [] * 2
        self.eyerightplace.append(0)
        self.eyerightplace.append(1)
        self.eyesightleft = [self.size, (self.size + 10)]
        self.eyesightright = [self.size, (self.size + 10)]
        self.seteyes()
        self.randmovevector()

    def saveLog(self, filename, strinki, fileaut):
        target = open(filename, fileaut)
        if (not isinstance(strinki, str)):
            for item in strinki:
                target.write("%s\n" % item)
        else:
            target.write(strinki)
        target.close()

    def seteyes(self):
        if(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] == 0):
            self.eyeleftplace[0] = self.X + self.size
            self.eyeleftplace[1] = self.Y - (self.size + 10)
            self.eyerightplace[0] = self.X + self.size
            self.eyerightplace[1] = self.Y + self.size
            self.eyesightleft = [self.size, (self.size + 10)]
            self.eyesightright = [self.size, (self.size + 10)]
        elif(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X + self.size
            self.eyeleftplace[1] = self.Y
            self.eyerightplace[0] = self.X
            self.eyerightplace[1] = self.Y + self.size
            self.eyesightleft = [(self.size + 10), self.size]
            self.eyesightright = [self.size, (self.size + 10)]
        elif(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X + self.size
            self.eyeleftplace[1] = self.Y + self.size
            self.eyerightplace[0] = self.X - (self.size + 10)
            self.eyerightplace[1] = self.Y + self.size
            self.eyesightleft = [(self.size + 10), self.size]
            self.eyesightright = [(self.size + 10), self.size]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] >= 1):
            self.eyeleftplace[0] = self.X
            self.eyeleftplace[1] = self.Y + self.size
            self.eyerightplace[0] = self.X - (self.size + 10)
            self.eyerightplace[1] = self.Y
            self.eyesightleft = [self.size, (self.size + 10)]
            self.eyesightright = [(self.size + 10), self.size]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] == 0):
            self.eyeleftplace[0] = self.X - self.size
            self.eyeleftplace[1] = self.Y + self.size
            self.eyerightplace[0] = self.X - self.size
            self.eyerightplace[1] = self.Y - (self.size + 10)
            self.eyesightleft = [self.size, (self.size + 10)]
            self.eyesightright = [self.size, (self.size + 10)]
        elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X
            self.eyeleftplace[1] = self.Y - (self.size + 10)
            self.eyerightplace[0] = self.X - (self.size + 10)
            self.eyerightplace[1] = self.Y
            self.eyesightleft = [self.size, (self.size + 10)]
            self.eyesightright = [(self.size + 10), self.size]
        elif(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X - (self.size + 10)
            self.eyeleftplace[1] = self.Y - self.size
            self.eyerightplace[0] = self.X + self.size
            self.eyerightplace[1] = self.Y - self.size
            self.eyesightleft = [(self.size + 10), self.size]
            self.eyesightright = [(self.size + 10), self.size]
        elif(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] <= -1):
            self.eyeleftplace[0] = self.X
            self.eyeleftplace[1] = self.Y - (self.size + 10)
            self.eyerightplace[0] = self.X + self.size
            self.eyerightplace[1] = self.Y
            self.eyesightleft = [self.size, (self.size + 10)]
            self.eyesightright = [(self.size + 10), self.size]

    def reinit(self):
        print("%s reinit" % (datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))))
        # self.pybrain_init(hidden_layers=random.randint(1, 40))
        self.randmovevector()
        self.seteyes()
        # self.nn.randomize()
        self.foodamount = 1
        self.X = random.randint(0, self.eartsize[0])
        self.Y = random.randint(0, self.eartsize[1])
        self.shadow[:] = []
        self.movecount = 0
        self.movememory = []
        # self.train()
        self.startime = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        print("%s reinit done" % (datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))))

    def live(self, dontPrintInfo=False, Learning=True):
        if self.move != "exception dood":
            try:
                hungri = 0
                foodtoeat = 0
                gotoleft = 0
                gotoright = 0
                if(self.foodamount <= 0.5):
                    hungri = 1
                if (self.foodavail >= 0.5):
                    foodtoeat = 1
                if (self.foodInLeft >= 0.5):
                    gotoleft = 1
                if (self.foodInRight >= 0.5):
                    gotoright = 1
                # trainingoutputs = self.gettraining([hungri, foodtoeat, gotoleft, gotoright])

                neuraloutputs = self.responce([hungri, self.foodavail, gotoleft, gotoright])

                if(dontPrintInfo is False):
                    # print ("IN", [hungri, foodtoeat, gotoleft, gotoright], "Tout:", self.roundfloat(trainingoutputs), "Nout:", self.roundfloat(neuraloutputs), [self.foodamount, self.foodavail, self.foodInLeft, self.foodInRight], self.nn.totalerror)
                    print ("IN", [hungri, foodtoeat, gotoleft, gotoright], "Nout:", self.roundfloat(neuraloutputs))

                if(round(neuraloutputs[0], 0) == 1):
                    if(self.foodavail >= 0.5):
                        if(dontPrintInfo is False):
                            print (self.printEpilogue, "eated", self.foodavail, "OK")

                    else:
                        if(dontPrintInfo is False):
                            print (self.printEpilogue, "eated", self.foodavail, "NOK")
                            print ("IN", [hungri, foodtoeat, gotoleft, gotoright], "Nout:", self.roundfloat(neuraloutputs))

                    self.foodamount = self.foodamount + self.foodavail - self.consumption  # - (neuraloutputs[1]/2)
                    self.eaten = "eated"
                    # print (self.printEpilogue, "eated", self.foodavail, "OK", self.foodamount)
                    # print("soi", round(neuraloutputs[0], 0) , neuraloutputs[0])
                else:
                    self.foodamount = self.foodamount - self.consumption  # - (neuraloutputs[1]/2)
                    self.eaten = "full"
                    # print("ei syoty")

                # self.foodamount = self.foodamount - self.consumption + (self.foodavail)  # - (neuraloutputs[1]/5)
                # set size based on energy
                if(self.foodamount < 0):
                    self.size = 2
                else:
                    self.size = int(self.foodamount * 6)
                    if(self.size < 2):
                        self.size = 2

                if(round(neuraloutputs[1], 0) == 1):
                    if(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] == 0):
                        self.randmovevector()
                    self.speed = neuraloutputs[1] * 3
                    self.X += self.directionvector[self.direction][0] * int(self.speed)
                    self.Y += self.directionvector[self.direction][1] * int(self.speed)
                    self.move = "moved"
                    self.seteyes()
                    self.shadow.append([self.X, self.Y])
                    if len(self.shadow) >= self.shadowlength:
                        del self.shadow[0]
                else:
                    self.speed = neuraloutputs[1]
                    self.move = "stopped"

                if(round(neuraloutputs[2], 0) == 1):  # move left
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
                    # if(dontPrintInfo == False):
                    #     print ("turn left", self.direction, self.directionvector[self.direction])
                elif(round(neuraloutputs[3], 0) == 1):  # move right
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
                    # if(dontPrintInfo == False):
                    #     print ("turn right", self.direction, self.directionvector[self.direction])

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

                if (self.movememory == self.roundfloat(neuraloutputs)):
                    if(dontPrintInfo is False):
                        print (self.movememory, self.movecount)
                else:
                    self.movememory = self.roundfloat(neuraloutputs)
                self.movecount += 1
                self.trainsteps += 1
            except:
                print ("Unexpected error:", sys.exc_info())
                self.eated = ""
                self.move = "exception dood"

        # print (self.roundfloat(trainingoutputs), self.roundfloat(neuraloutputs), self.roundfloat([self.foodamount, self.foodavail]))
    def didoueat(self):
        if("eated" in self.eaten):
            # print(self.filename, "eaten")
            self.eaten = "full"
            return True
        else:
            return False

    def addfoodavail(self, addfood):
        self.foodavail = addfood

    def turnleft(self, foodInLeft):
        self.foodInLeft = foodInLeft

    def turnright(self, foodInRight):
        self.foodInRight = foodInRight

    def randmovevector(self):
        vectorok = 1
        temp = random.randint(0, 7)
        while(vectorok):
            if(temp == self.direction):
                temp = random.randint(0, 7)
            else:
                self.direction = temp
                vectorok = 0

    def roundfloat(self, rounuppilist):
        roundedlist = []
        for i in range(len(rounuppilist)):
            roundedlist.append(round(rounuppilist[i], 0))
        return roundedlist

    def getliveinfo(self):
        # time2 = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        # diff = time2 - self.startime
        return [round(self.foodamount, 4), self.filename.split('.')[0], self.movecount]
        # return [round(self.foodamount, 4), round(self.speed, 4), self.filename, diff.total_seconds(), self.movecount]

    def areyouallive(self):
        time2 = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        diff = time2 - self.startime
        if(self.test):
            if(self.foodamount < -5.00 or self.foodamount > 5.00):
                return (["dood", self.foodamount, self.move])
            elif(diff.total_seconds() > 30):
                # self.saveLog(self.filename, self.nn.inspectTofile(), 'a+')
                return (["viable NN"])
            if(self.move == "exception dood"):
                return (["dood", self.move])
            return ["ok"]
        else:
            if(self.foodamount < -5.00 or self.foodamount > 5.00):
                    if (self.foodamount > 5.00):
                        # print ("Not viable motko, randomize %s" % (self.getliveinfo()))
                        # self.reinit()
                        return ["dood"]
                    else:
                        # print ("Not viable motko, randomize %s" % (self.getliveinfo()))
                        # self.reinit()
                        return ["dood"]
            elif(diff.total_seconds() > 900):
                # self.saveViableNN()
                # self.saveLog(self.filename, self.nn.inspectTofile(), 'a+')
                return (["viable NN"])
            if(self.move == "exception dood"):
                return (["dood", self.move])
            return ["ok"]

    def getname(self):
        return self.filename

    def setname(self, name):
        self.filename = name

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

    def leftorright(left, right):
        if(left > right):
            return 0
        elif(left <= right):
            return 1
        elif(left == 0.01 and right == 0.01):
            return 2

    def eat(self, inputs):
        # EAT energy, food avail, food left, food right, food color, color,
        outputs = [1, 0, 0, 0, 0]  # default response
        if(inputs[0] < 0.75):  # hungry
            if(inputs[1] < inputs[2] or inputs[4] == inputs[5]):  # food in left more than front and food color is different than in your color meaning eatable
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 1  # move
                outputs[3] = 1  # turn left
                outputs[4] = 0  # do not turn right
            elif(inputs[1] < inputs[3] or inputs[4] == inputs[5]):  # food in right more than front
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 1  # move
                outputs[3] = 0  # do not turn left
                outputs[4] = 1  # turn right
            elif(inputs[4] == inputs[5]):  # food is same color dont eat it will kill you
                outputs[0] = 0  # dont eat
                outputs[1] = 0  # dont eat anything
                outputs[2] = 1  # move
                outputs[3] = 1  # turn right we prefer left
                outputs[4] = 0  # do not turn right
            elif(inputs[4] != inputs[5]):  # food is different color than you, it is eatable
                if((inputs[0] + inputs[1]) < 1.5):  # it is not too mutch
                    outputs[0] = 1  # dont eat
                    outputs[1] = inputs[1]  # eat all
                    outputs[2] = 1  # move
                    outputs[4] = 0  # turn right
                    outputs[3] = 0  # do not turn right
                    outputs[4] = 0.25  # slow down food front of you
                else:
                    outputs[0] = 1  # dont eat
                    outputs[1] = (inputs[0] + inputs[1]) - 1  # dont eat anything
                    outputs[2] = 1  # move
                    outputs[3] = 0  # turn right
                    outputs[4] = 0  # do not turn right
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

    def contact(self, inputs):
        # if same color as you then sex or flee if different color flee or kill. by killing you get enegry what other has
        outputs = [0, 0, 0]  # default response
        if(inputs[5] == inputs[6]):  # same so flee or sex, energyamount defines what
            if(inputs[0] > 0.50):  # enought food to sex
                outputs[8] = inputs[0]
            else:
                outputs[7] = 1 - inputs[0]  # less amount food more fleeing
        else:
            if(inputs[0] > 0.50):  # enought food to flee
                outputs[7] = inputs[0]
            else:  # hungry so fight
                inputs[6] = 1 - inputs[0]

        return outputs

    def gettraining2(self, inputs):
        # inputs are: energy 0, food avail 1, food left 2, food right 3, food color 4, color 5, meeting motko color6,
        # outputs are: eat 0, eat amount 1, move 2, speed 3, turn left 4, turn tight 5, kill 6, flee 7, sex 8
        # divide trainign to smaller parts
        eatoutputs = eat(inputs)  # eat, eat amount, move, turn left, turn tight
        print (eatoutputs)
        contactouputs = contact(inputs)
        print (contactouputs)
        return eatoutputs + contactouputs
