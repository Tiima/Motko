# !/usr/bin/python
import os
import sys
import foodblock
import random
import motko
import pygame
import logging
from common import timing_function
import argparse


@timing_function
def loadmotkos(path, amount, trainingloops, size, hiddenlayers, loadfromfile, test):
    loadedmotkos = []
    motkoslist = os.listdir(os.path.join(path, 'brains'))

    for k in range(len(motkoslist)):
        if(int(motkoslist[k].split('_')[1]) == trainingloops):
            motkoinstance = motko.motkowrapper(os.path.join(path, 'brains', motkoslist[k]), size, hiddenlayers, loadfromfile, test)
            motkoinstance.motkolive.setname(motkoslist[k])
            print("loading motko", motkoinstance.motkolive.getname())
            motkoinstance.motkolive.reinit()
            # if(test):
            #    os.remove(os.path.join(path, 'brains', motkoslist[k]))
            loadedmotkos.append(motkoinstance)
            if(len(loadedmotkos) >= amount):
                return loadedmotkos
    return loadedmotkos


class PyManMain:
    """The Main PyMan Class - This class handles the main
    initialization and creating of the Game."""

    @timing_function
    def __init__(self, test, noscreen, width=1024, height=768, foodamount=400, motkotamount=20):
        """Initialize"""
        self.cwd = os.getcwd()
        logging.basicConfig(filename=os.path.join(self.cwd, "pygame_main.log"), filemode='w', level=logging.INFO)
        logging.info("start main")
        self.gamescreen = True
        self.test = False
        self.motkotamount = motkotamount
        self.test = test
        if(noscreen):
            self.gamescreen = False
            print ("setted", sys.argv[1], self.gamescreen)
        if(test):
            self.gamescreen = False
            print ("setted", sys.argv[1], self.gamescreen)
            self.test = True
            self.motkotamount = 3
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
        self.trainingloops = 1
        self.hiddenlayers = 4
        self.SaveWhenTrained = 100
        # self.sleeptime = 0.1

        if(self.gamescreen):
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.screen2 = pygame.display.set_mode((self.width, self.height))
            self.myfont = pygame.font.SysFont("monospace", 15)

        for i in range(self.foodamount):
            tempplace = [random.randint(0, self.width), random.randint(0, self.height)]
            fb = foodblock.foodblock(tempplace, [2, 2])
            # print (fb.getinfo())
            self.foodblocks.append(fb)
        self.cwd = os.getcwd()
        self.motkot = loadmotkos(self.cwd, self.motkotamount, self.trainingloops, [self.width, self.height], self.hiddenlayers, loadfromfile=True, test=self.test)

    @timing_function
    def MainLoop(self):
        """This is the Main Loop of the Game"""
        if(self.gamescreen):
            pygame.key.set_repeat(500, 30)
            self.background = pygame.Surface(self.screen.get_size())
            self.background = self.background.convert()
            self.background.fill((255, 255, 255))
        print ("mainloop testing %s motkos at same time" % (self.motkotamount))
        dontprintdata = False
        printsnapshotinfo = False
        last_steps = []
        self.FoodBlockID = [] * self.motkotamount
        for _ in range(self.motkotamount):
            self.FoodBlockID.append(None)
        print len(self.FoodBlockID)
        print len(self.motkot)
        printmotko = 99
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
                        elif (event.key == pygame.K_UP):
                            printsnapshotinfo = True
                        elif (event.key == pygame.K_DOWN):
                            printmotko = 99
                        elif (event.key == pygame.K_1):
                            printmotko = 1
                        elif (event.key == pygame.K_2):
                            printmotko = 2
                        elif (event.key == pygame.K_3):
                            printmotko = 3
                        elif (event.key == pygame.K_4):
                            printmotko = 4
                        elif (event.key == pygame.K_5):
                            printmotko = 5
                        elif (event.key == pygame.K_6):
                            printmotko = 6
                        elif (event.key == pygame.K_7):
                            printmotko = 7
                        elif (event.key == pygame.K_8):
                            printmotko = 8
                        elif (event.key == pygame.K_9):
                            printmotko = 9
                        elif (event.key == pygame.K_0):
                            printmotko = 0
            for k in range(len(self.motkot) - 1, -1, -1):
                # print (self.motkot[k].motkolive.returnname())
                status = self.motkot[k].motkolive.areyouallive()
                if status[0] == "dood" or status[0] == "viable NN":
                    last_steps = self.motkot[k].motkolive.getliveinfo()
                    self.motkot[k].setNotViable()
                    print ("!:", self.motkot[k].motkolive.getname(), status, self.motkot[k].motkolive.getliveinfo())
                    # print (self.motkot[k].motkolive.nn.inspect())
                    del self.motkot[k]
                # for i in range(len(self.motkot) - 1, -1, -1):
                # if (self.motkot[k].motkolive.collision([self.motkot[k].motkolive.X, self.motkot[k].motkolive.Y], [self.motkot[k].motkolive.size, self.motkot[k].motkolive.size]) == 1):
                # print("kill or sex or flee?")
            textplaceY = 0

            for k in range(len(self.motkot)):
                # self.motkot[k].motkolive.addfoodavail(0.001)
                for i in range(len(self.motkot[k].motkolive.shadow)):
                    if(self.gamescreen):
                        pygame.draw.rect(self.screen, [255 - (i * 2), 255 - (i * 2), 255 - (i * 2)], [self.motkot[k].motkolive.shadow[i][0], self.motkot[k].motkolive.shadow[i][1], 2, 2], 0)
                # print ("check food bloks")
                for i in range(len(self.foodblocks)):
                    if self.foodblocks[i].collision([self.motkot[k].motkolive.X, self.motkot[k].motkolive.Y], [self.motkot[k].motkolive.size, self.motkot[k].motkolive.size]) == 1:
                        self.motkot[k].motkolive.addfoodavail(self.foodblocks[i].returnfoodamount(), self.foodblocks[i].colornumber)
                        self.FoodBlockID[k] = i
                    if(self.foodblocks[i].collision([self.motkot[k].motkolive.eyeleftplace[0], self.motkot[k].motkolive.eyeleftplace[1]], self.motkot[k].motkolive.eyesightsizeleft) == 1):
                        self.motkot[k].motkolive.foodleft(self.foodblocks[i].returnfoodamount())
                        # print ("left hit!")
                    if(self.foodblocks[i].collision([self.motkot[k].motkolive.eyerightplace[0], self.motkot[k].motkolive.eyerightplace[1]], self.motkot[k].motkolive.eyesightsizeright) == 1):
                        self.motkot[k].motkolive.foodright(self.foodblocks[i].returnfoodamount())
                        # print ("right hit!")

                if(printmotko != 99):
                    dontprintdata = False
                    if(k == printmotko):
                        self.motkot[k].motkolive.live(True, test=self.test)
                    else:
                        self.motkot[k].motkolive.live(False, test=self.test)
                else:
                    self.motkot[k].motkolive.live(dontprintdata, test=self.test)
                didoueat = self.motkot[k].motkolive.didoueat()
                if(len(self.FoodBlockID) > 0):
                    if(didoueat > 0 and self.FoodBlockID[k] is not None):
                        self.foodblocks[self.FoodBlockID[k]].calculateFoodamount(didoueat)
                        if (self.foodblocks[self.FoodBlockID[k]].foodamount <= 0.01):
                            del self.foodblocks[self.FoodBlockID[k]]
                            tempplace = [random.randint(0, self.width), random.randint(0, self.height)]
                            newfb = foodblock.foodblock(tempplace, [2, 2])
                            self.FoodBlockID[k] = None
                            self.foodblocks.append(newfb)

                if(self.motkot[k].motkolive.trainings == self.SaveWhenTrained):
                    name = "%s_%s_%s.%d.pkl" % (self.motkot[k].motkolive.filename.split('_')[0], self.SaveWhenTrained, self.motkot[k].motkolive.filename.split('_')[2])
                    self.motkot[k].saveNNwithname(name)
                    self.SaveWhenTrained += 100
                    print("Saving {} with errorcount {}".format(name, self.motkot[k].motkolive.currenterror))

                if(printsnapshotinfo):
                    print(self.motkot[k].motkolive.getliveinfo2())

                if(self.gamescreen):
                    pygame.draw.rect(self.screen, self.motkot[k].motkolive.color, [self.motkot[k].motkolive.X, self.motkot[k].motkolive.Y, self.motkot[k].motkolive.size, self.motkot[k].motkolive.size], 0)
                    pygame.draw.rect(self.screen, self.motkot[k].motkolive.color, [self.motkot[k].motkolive.eyeleftplace[0], self.motkot[k].motkolive.eyeleftplace[1], self.motkot[k].motkolive.eyesightsizeleft[0], self.motkot[k].motkolive.eyesightsizeleft[1]], 1)
                    pygame.draw.rect(self.screen, self.motkot[k].motkolive.color, [self.motkot[k].motkolive.eyerightplace[0], self.motkot[k].motkolive.eyerightplace[1], self.motkot[k].motkolive.eyesightsizeright[0], self.motkot[k].motkolive.eyesightsizeright[1]], 1)
                    if (len(self.motkot) < 6):
                        coretext = self.myfont.render(str(self.motkot[k].motkolive.getliveinfo()), 1, (0, 0, 0), (255, 255, 255))
                        self.screen.blit(coretext, (0, textplaceY))
                        textplaceY += 15
                # else:
                    # print(self.motkot[k].motkolive.getliveinfo())
            for i in range(len(self.foodblocks)):
                if(self.gamescreen):
                    pygame.draw.rect(self.screen, self.foodblocks[i].color, [self.foodblocks[i].X, self.foodblocks[i].Y, self.foodblocks[i].getSize1(), self.foodblocks[i].getSize2()], 0)

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
                # time.sleep(self.sleeptime)
            if (len(self.motkot) == 0):
                if(len(last_steps) > 2):
                    print ("last motko: ", last_steps[1], "trained:", last_steps[1].split('_')[1], "times, steps taken", last_steps[2], last_steps[0])
                    last_steps = []
                self.trainingloops += 1
                self.motkot = loadmotkos(self.cwd, self.motkotamount, self.trainingloops, [self.width, self.height], self.hiddenlayers, loadfromfile=True, test=self.test)

            if(self.trainingloops > 1000000):  # i think that is enought
                break
            if(self.test and self.trainingloops > 1000):
                break
            printsnapshotinfo = False

if __name__ == "__main__":
    logging.basicConfig(filename="pygame_main.log", format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.info("pygame_main start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="run testrun", action="store_true")
    parser.add_argument("--width", help="Width of the world, default 1024", default=1024, type=int)
    parser.add_argument("--height", help="Height of the world, default 768", default=768, type=int)
    parser.add_argument("--foodamount", help="food amount in world, default 400", default=400, type=int)
    parser.add_argument("--motkotamount", help="How many motkos you have in your world, default 3", default=3, type=int)
    parser.add_argument("--noscreen", help="run headles", action="store_true")
    args = parser.parse_args()
    logging.info(args)
    MainWindow = PyManMain(test=args.test, width=args.width, height=args.height, foodamount=args.foodamount, motkotamount=args.motkotamount, noscreen=args.noscreen)
    MainWindow.MainLoop()
