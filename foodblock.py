# !/usr/bin/python
import random


class foodblock:

    def __init__(self, place=[0, 0], size=[2, 2]):
        self.X = place[0]
        self.Y = place[1]
        self.size = size
        self.foodamount = random.uniform(0.5, 0.99)
        self.size[0] = self.size[0] + int(self.foodamount * 5)
        self.size[1] = self.size[1] + int(self.foodamount * 5)

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
