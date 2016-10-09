# !/usr/bin/python
import os
import sys
import time
import foodblock
import random
import motko
import multiprocessing
import pygame


def trainer(name, size, hiddenlayer):
    # self.threadLock.acquire()
    print ("Start training ", name, size, hiddenlayer)
    motkoinstance = motko.motko(name, size, hiddenlayer)
    # self.threadLock.release()
    motkoinstance.train()  # (10000*(hiddenneuron*hiddenlayer)))
    motkoinstance.saveNN()


class PyManMain:
    """The Main PyMan Class - This class handles the main
    initialization and creating of the Game."""

    def __init__(self, width=1024, height=768, foodamount=600, motkotamount=20):
        """Initialize"""
        self.gamescreen = True
        self.test = False
        self.motkotamount = motkotamount
        if (len(sys.argv) == 2):
            if("no" in sys.argv[1]):
                self.gamescreen = False
                print ("setted", sys.argv[1], self.gamescreen)
                self.motkotamount = 10
            elif("test" in sys.argv[1]):
                self.gamescreen = False
                print ("setted", sys.argv[1], self.gamescreen)
                self.test = True
                self.motkotamount = 2
        if(self.gamescreen):
            pygame.init()
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.RED2 = (242, 120, 120)
        self.PURB = (255, 0, 255)
        self.GRAY = (224, 224, 224)
        self.GRAY2 = (192, 192, 192)
        self.GRAY3 = (160, 160, 160)
        self.GRAY4 = (128, 128, 128)
        self.GRAY5 = (96, 96, 96)
        self.GRAY6 = (32, 32, 32)
        self.X = 0
        self.Y = 0
        self.width = width
        self.height = height
        self.foodamount = foodamount
        self.foodblocks = []

        self.sleeptime = 0.1
        if(self.gamescreen):
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.screen2 = pygame.display.set_mode((self.width, self.height))
            self.myfont = pygame.font.SysFont("monospace", 15)
        self.stoplayer = 60
        if(self.test):
            self.stoplayer = 9
        self.hiddenlayerstart = 2
        self.hiddenlayer = 2
        if(self.gamescreen):
            self.maxtrainers = 3
        else:
            self.maxtrainers = 4

        for i in range(self.foodamount):
            tempplace = [random.randint(0, self.width), random.randint(0, self.height)]
            fb = foodblock.foodblock(tempplace, [2, 2])
            # print (fb.getinfo())
            self.foodblocks.append(fb)

        self.motkot = []
        self.cwd = os.getcwd()
        trainers = []

        print ("creating and training NNs start layer %s stop layeramount %s" % (self.hiddenlayer, self.stoplayer))

        # trainer(("motko_%d"%(self.hiddenlayer)), [self.width, self.height], self.hiddenlayer)
        childthreads = 0

        while 1:
            if(os.path.isfile(os.path.join(self.cwd, 'brains', "motko_%d.pybrain_pkl" % (self.hiddenlayer))) is False):
                print ("training motko_%d.pybrain_pkl" % (self.hiddenlayer))
                p = multiprocessing.Process(target=trainer, args=(("motko_%d" % (self.hiddenlayer)), [self.width, self.height], self.hiddenlayer))
                trainers.append(p)
                p.start()
                childthreads += 1
                # trainer(("motko_%d"%(self.hiddenlayer)), [self.width, self.height], self.hiddenlayer)
                # motko = motko.motko(("motko_%d_%d"%(self.hiddenlayer, self.hiddenneuron)), [self.width, self.height], self.hiddenneuron, self.hiddenlayer)
                # motko.train(True, True, 10000)
                # motko.saveNN()
                if(childthreads == self.maxtrainers):
                    print("waiting trainers")
                    for t in trainers:
                        # print ('t.is_alive()', t.is_alive())
                        t.join()
                    childthreads = 0
            else:
                print ("file motko_%d.pkl exist" % (self.hiddenlayer))
            self.hiddenlayer += 1
            if (self.hiddenlayer > self.stoplayer):
                break

        print ("waiting threads to finish")
        for t in trainers:
            # print ('t.is_alive()', t.is_alive())
            t.join()

        print ("creating and training NNs done")

        self.hiddenlayer = self.hiddenlayerstart

        while(True):
            if(os.path.isfile(os.path.join(self.cwd, 'brains', ("motko_%d.pybrain_pkl.pkl_noviable" % (self.hiddenlayer)))) is False and os.path.isfile(os.path.join(self.cwd, 'brains', ("motko_%d.pybrain_pkl" % (self.hiddenlayer)))) is True):
                motkoinstance = motko.motko(("motko_%d" % (self.hiddenlayer)), [self.width, self.height], self.hiddenlayer, loadfromfile=True, test=self.test)
                self.motkot.append(motkoinstance)
            else:
                print ("Skip motko_%d.pybrain_pkl.pkl_noviable" % (self.hiddenlayer))
            # motko.train()
            print (len(self.motkot), self.motkotamount)
            if(len(self.motkot) >= self.motkotamount):
                break
            if(self.hiddenlayer > self.stoplayer):
                break
            self.hiddenlayer += 1

    def MainLoop(self):
        """This is the Main Loop of the Game"""
        if(self.gamescreen):
            pygame.key.set_repeat(500, 30)
            self.background = pygame.Surface(self.screen.get_size())
            self.background = self.background.convert()
            self.background.fill((0, 0, 0))
        print ("mainloop testing %s motkos at same time" % (self.motkotamount))
        dontprintdata = True
        step = True
        # deletemotkoindex = []

        while 1:
            # loopStart = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
            if(self.gamescreen):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if (event.key == pygame.K_RIGHT):
                            dontprintdata = False
                        elif (event.key == pygame.K_LEFT):
                            dontprintdata = True
                            print(len(self.motkot))
                        elif (event.key == pygame.K_UP):
                            # for k in range(self.motkotamount):
                            #     print (self.motkot[k].nn.inspect())
                            self.sleeptime += 0.01
                            print (self.sleeptime)
                        elif (event.key == pygame.K_DOWN):
                            if(self.sleeptime <= 0.01):
                                self.sleeptime = 0.01
                            else:
                                self.sleeptime -= 0.01
                            print (self.sleeptime)
                            # if(step == True):
                            #     step = False
                            # else:
                            #     step = True
            # print (len(self.motkot), self.hiddenlayer)
            for k in range(len(self.motkot) - 1, -1, -1):
                # print (self.motkot[k].returnname())
                status = self.motkot[k].areyouallive()
                if status[0] == "dood" or status[0] == "icanseemyhousefromhere":
                    print ("!:", self.motkot[k].returnname(), status, self.motkot[k].getliveinfo())
                    # print (self.motkot[k].nn.inspect())
                    del self.motkot[k]
                    # deletemotkoindex.append(k)
                    if (self.hiddenlayer != self.stoplayer):  # no more new motkos
                        self.hiddenlayer += 1
                        if(os.path.isfile(os.path.join(self.cwd, 'brains', ("motko_%d.pybrain_pkl" % (self.hiddenlayer)))) is True):
                            if(os.path.isfile(os.path.join(self.cwd, 'brains', ("motko_%d.pybrain_pkl.pkl_noviable" % (self.hiddenlayer)))) is False):
                                motkoinstance = motko.motko(("motko_%d" % (self.hiddenlayer)), [self.width, self.height], self.hiddenlayer, loadfromfile=True, test=self.test)
                                self.motkot.append(motkoinstance)
                                print ("appended motko motko_%d" % (self.hiddenlayer))
                            else:
                                print ("Skip motko_%d.pybrain_pkl.pkl_noviable" % (self.hiddenlayer))
            textplaceY = 0

            for k in range(len(self.motkot)):
                # self.motkot[k].addfoodavail(0.001)
                self.motkot[k].turnleft(0)
                self.motkot[k].turnright(0)
                for i in range(len(self.motkot[k].shadow) - 1, 0, -1):
                    if(self.gamescreen):
                        pygame.draw.rect(self.screen, [i * 2, i * 2, i * 2, ], [self.motkot[k].shadow[i][0], self.motkot[k].shadow[i][1], 5, 5], 0)

                for i in range(len(self.foodblocks)):
                    # print (self.motkot[k].X, self.motkot[k].Y)
                    if self.foodblocks[i].collision([self.motkot[k].X, self.motkot[k].Y], [5, 5]) == 1:
                        # print ("motko soi", self.foodblocks[i].getinfo())
                        self.motkot[k].addfoodavail(self.foodblocks[i].returnfoodamount())
                        # print ("eated2222", self.motkot[k].printEpilogue, self.foodblocks[i].returnfoodamount())
                        # print("foodavail", self.motkot[k].printEpilogue, self.motkot[k].foodavail)
                        if(self.motkot[k].didoueat()):
                            del self.foodblocks[i]
                            tempplace = [random.randint(0, self.width), random.randint(0, self.height)]
                            newfb = foodblock.foodblock(tempplace, [2, 2])
                            # print (newfb.getinfo())
                            self.foodblocks.append(newfb)
                    else:
                        if(self.gamescreen):
                            pygame.draw.rect(self.screen, self.GREEN, [self.foodblocks[i].X, self.foodblocks[i].Y, self.foodblocks[i].size[0], self.foodblocks[i].size[1]], 0)
                    if(self.foodblocks[i].collision([self.motkot[k].eyeleftplace[0], self.motkot[k].eyeleftplace[1]], self.motkot[k].eyesightleft) == 1):
                        self.motkot[k].turnleft(self.foodblocks[i].returnfoodamount())
                        # print ("left hit!")
                    if(self.foodblocks[i].collision([self.motkot[k].eyerightplace[0], self.motkot[k].eyerightplace[1]], self.motkot[k].eyesightright) == 1):
                        self.motkot[k].turnright(self.foodblocks[i].returnfoodamount())
                        # print ("right hit!")

                if (step):
                    self.motkot[k].live(dontprintdata, False)
                    # step = False
                if(self.gamescreen):
                    pygame.draw.rect(self.screen, self.RED, [self.motkot[k].X, self.motkot[k].Y, 5, 5], 0)
                    pygame.draw.rect(self.screen, self.PURB, [self.motkot[k].eyeleftplace[0], self.motkot[k].eyeleftplace[1], self.motkot[k].eyesightleft[0], self.motkot[k].eyesightleft[1]], 1)
                    pygame.draw.rect(self.screen, self.BLUE, [self.motkot[k].eyerightplace[0], self.motkot[k].eyerightplace[1], self.motkot[k].eyesightright[0], self.motkot[k].eyesightright[1]], 1)
                    # print (self.motkot[k].X, self.motkot[k].Y, self.motkot[k].eyeleft[0], self.motkot[k].eyeleft[1], self.motkot[k].eyesightleft[0], self.motkot[k].eyesightleft[1])
                    if (self.motkotamount < 11):
                        coretext = self.myfont.render(str(self.motkot[k].getliveinfo()), 1, (255, 255, 255), (0, 0, 0))
                        self.screen.blit(coretext, (0, textplaceY))
                        textplaceY += 15
                # else:
                    # print(self.motkot[k].getliveinfo())
            if(self.gamescreen):
                for i in range(0, self.width, 25):
                    for k in range(0, self.height, 25):
                        pygame.draw.rect(self.screen, self.GRAY, [i, k, 1, 1], 1)

            # pygame.draw.rect(self.screen, self.GREEN, [self.X, self.Y, 5, 5], 0)
            if(self.gamescreen):
                pygame.display.flip()
                self.screen.blit(self.background, (0, 0))
                # oopStop = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
                # delta = loopStop - loopStart
                # print (self.sleeptime, delta.total_seconds(), (delta.total_seconds()/1000), delta, (self.sleeptime-(delta.total_seconds()/1000)))
                # time.sleep(self.sleeptime-(delta.total_seconds()/1000))
                time.sleep(self.sleeptime)
            if (self.hiddenlayer > self.stoplayer):
                break
            if (len(self.motkot) == 0):
                break
                # screen2 print one motko

if __name__ == "__main__":
        MainWindow = PyManMain()
        MainWindow.MainLoop()
