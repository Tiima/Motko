# !/usr/bin/python
import random


class foodblock:

    def __init__(self, place=[0, 0], size=[2, 2]):
        self.X = place[0]
        self.Y = place[1]
        self.size = size
        # because threading
        random.jumpahead(1252157)
        self.foodamount = random.random()
        self.size[0] = self.size[0] + int(self.foodamount * 6)
        self.size[1] = self.size[1] + int(self.foodamount * 6)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        colors = []
        colors.append(self.RED)
        colors.append(self.BLACK)
        colors.append(self.GREEN)
        colors.append(self.BLUE)
        self.colornumber = random.randint(0, 3)
        self.color = colors[self.colornumber]
        # TODO create color to the food

    def collision(self, collider, collidersize):
        if (int(self.X) < int(collider[0]) + int(collidersize[0]) and int(self.X) + int(self.size[0]) > int(collider[0]) and int(self.Y) < int(collider[1]) + int(collidersize[1]) and int(self.size[1]) + int(self.Y) > int(collider[1])):
            # print (collider, self.X, self.Y)
            return 1  # collision
        else:
            return 0

    def getinfo(self):
        return [self.X, self.Y, self.foodamount]

    def returnfoodamount(self):
        return self.foodamount

    def calculateFoodamount(self, eated):
        self.foodamount = self.foodamount - eated
        if(self.foodamount > 0):
            self.size[0] = self.size[0] + int(self.foodamount * 6)
            self.size[1] = self.size[1] + int(self.foodamount * 6)

    def getSize1(self):
        return self.size[0]

    def getSize2(self):
        return self.size[1]
