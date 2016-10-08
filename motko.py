#!/usr/bin/python

import os, sys
import time
import operator
import foodblock
import random
import datetime
import multiprocessing
import pickle
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
import pybrain

class motko:

	def gettraining(self, inputs):
		#print ("calculated_output", calculated_output)
		training_outputs = [] * 4
		#print ("gettraining, inputs", inputs)
		if(inputs == [1,0,0,0]): #full
			training_outputs.append(0) 
			training_outputs.append(1) #move
			training_outputs.append(0)
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [1,1,0,0]): #hungri foo avail
			training_outputs.append(1)  #eat
			training_outputs.append(0)
			training_outputs.append(0)
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [1,1,1,0]): #hungri foodavail food in left
			training_outputs.append(1) #eat
			training_outputs.append(0)
			training_outputs.append(1) #turn
			training_outputs.append(0) 
			return training_outputs
		elif(inputs == [1,1,1,1]): # hungri foodavail food in left and right
			training_outputs.append(1) #eat
			training_outputs.append(0)
			training_outputs.append(1) # prefer left
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [1,1,0,1]):  # hungri foodavail food in right
			training_outputs.append(1) #eat
			training_outputs.append(0)
			training_outputs.append(0)
			training_outputs.append(1) #turn
			return training_outputs
		elif(inputs == [0,1,1,1]): #full foodavail food in left and right
			training_outputs.append(0) 
			training_outputs.append(0) 
			training_outputs.append(0)
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [0,0,1,1]): #full food in left and right
			training_outputs.append(0) 
			training_outputs.append(0) #should we move?
			training_outputs.append(1) # prefer left
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [0,0,0,1]):  #food in right
			training_outputs.append(0) 
			training_outputs.append(0)
			training_outputs.append(0)
			training_outputs.append(1) #turn
			return training_outputs
		elif(inputs == [1,0,1,0]): # hungri food in left
			training_outputs.append(0) 
			training_outputs.append(1) #move
			training_outputs.append(1) #turn
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [1,0,0,1]):  #hungri food in right
			training_outputs.append(0) 
			training_outputs.append(1) #move
			training_outputs.append(0)
			training_outputs.append(1) #turn
			return training_outputs
		elif(inputs == [1,0,1,1]): #hungri food in left and right
			training_outputs.append(0)
			training_outputs.append(1) #move
			training_outputs.append(1) # we prefer left
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [0,1,0,0]): #full foodavail
			training_outputs.append(0) 
			training_outputs.append(1) # should we prefer stop and wait until hungri again?
			training_outputs.append(0)
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [0,0,1,0]): #full food in left
			training_outputs.append(0) 
			training_outputs.append(1) #move
			training_outputs.append(1) #tur
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [0,0,0,0]): #full no foods
			training_outputs.append(0) 
			training_outputs.append(1) #move
			training_outputs.append(0)
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [0,1,0,1]): #foodavail food in right
			training_outputs.append(0) 
			training_outputs.append(1)
			training_outputs.append(0)
			training_outputs.append(1)
			return training_outputs
		elif(inputs == [0,1,1,0]): 
			training_outputs.append(0) 
			training_outputs.append(1) #move
			training_outputs.append(1) #we prefer left
			training_outputs.append(0)
			return training_outputs
		elif(inputs == [0,0,0,0]): #hungri foo avail
			training_outputs.append(0)  #eat
			training_outputs.append(1)
			training_outputs.append(0)
			training_outputs.append(0)
			return training_outputs
		else:
			print ("fail!!!", inputs)

	def pybrain_init(self, input_amount=4, output_amount=4, hidden_layers=4): 
		ds = SupervisedDataSet(input_amount, output_amount)
		for i in range(16):
			trainingsetup = format(i, '04b')
			trainingresult = self.gettraining([int(trainingsetup[0]),int(trainingsetup[1]),int(trainingsetup[2]),int(trainingsetup[3])])
			#print ([int(trainingsetup[0]),int(trainingsetup[1]),int(trainingsetup[2]),int(trainingsetup[3])], trainingresult)
			ds.addSample((int(trainingsetup[0]),int(trainingsetup[1]),int(trainingsetup[2]),int(trainingsetup[3])), 
				         (trainingresult[0],trainingresult[1],trainingresult[2],trainingresult[3]))

		self.nn = buildNetwork(4, hidden_layers, 4, bias=True, hiddenclass=TanhLayer)
		self.trainer = BackpropTrainer(self.nn, ds)

	def train(self):
		for _ in range(100):
			#self.trainer.train()
			self.trainer.trainUntilConvergence(validationProportion=0.1)

	def responce(self,liveinput):
		#ds = SupervisedDataSet(4, 4)
		"""for i in range(16):
			trainingsetup = format(i, '04b')
			trainingresult = self.gettraining([int(trainingsetup[0]),int(trainingsetup[1]),int(trainingsetup[2]),int(trainingsetup[3])])
			#print ([int(trainingsetup[0]),int(trainingsetup[1]),int(trainingsetup[2]),int(trainingsetup[3])], trainingresult)
			ds.addSample((int(trainingsetup[0]),int(trainingsetup[1]),int(trainingsetup[2]),int(trainingsetup[3])), 
				         (trainingresult[0],trainingresult[1],trainingresult[2],trainingresult[3]))"""

		"""self.trainer = BackpropTrainer(self.nn, ds)"""
		#self.trainer.train()
		#self.trainer.trainUntilConvergence(validationProportion=0.1)
		return self.nn.activate((liveinput[0],liveinput[1],liveinput[2],liveinput[3]))

	def __init__(self,filename, eartsize, num_hiddeLayers="FF", loadfromfile=False):
		self.cwd = os.getcwd()
		self.filename = filename
		self.objectname = filename
		self.printEpilogue = filename 
		self.objectname += ".pybrain_pkl"
		self.filename+=".log"
		if(num_hiddeLayers == "FF"):
			num_hiddeLayers = random.randint(2,20)
		if(loadfromfile == True):
			print (self.objectname)
			self.nn = pickle.load(open(os.path.join(self.cwd,'brains',self.objectname), "rb" ))
		else:
			self.pybrain_init(hidden_layers=num_hiddeLayers)
		print (filename, num_hiddeLayers)
		self.foodamount = 0.9
		self.eartsize = eartsize
		self.X = random.randint(0,eartsize[0])
		self.Y = random.randint(0,eartsize[1])

		self.consumption = 0.05
		self.foodavail = 0
		self.shadow = []
		self.shadowlength = 100
		self.startime = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
		self.move =  "moved"
		self.eaten = "full"
		self.speed = int(1)
		self.foodInLeft = 0
		self.eyeleftplace = []*2
		self.eyeleftplace.append(0)
		self.eyeleftplace.append(1)
		self.foodInRight = 0
		self.eyerightplace = []*2
		self.eyerightplace.append(0)
		self.eyerightplace.append(1)
		self.eyesightleft = [5,15]
		self.eyesightright = [5,15]
		#print (self.X, self.Y)
		self.direction = 0
		self.directionvector = [] * 8
		self.directionvector.append([1,0])
		self.directionvector.append([1,1])
		self.directionvector.append([0,1])
		self.directionvector.append([-1,1])
		self.directionvector.append([-1,0])
		self.directionvector.append([-1,-1])
		self.directionvector.append([0,-1])
		self.directionvector.append([1,-1])
		self.seteyes()
		self.randmovevector()
		self.movecount = 0
		self.movememory = []

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
			self.eyeleftplace[0] = self.X + 5
			self.eyeleftplace[1] = self.Y - 15
			self.eyerightplace[0] = self.X + 5
			self.eyerightplace[1] = self.Y + 5
			self.eyesightleft=[5,15]
			self.eyesightright=[5,15]
		elif(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] >= 1):
			self.eyeleftplace[0] = self.X + 5
			self.eyeleftplace[1] = self.Y 
			self.eyerightplace[0] = self.X
			self.eyerightplace[1] = self.Y + 5
			self.eyesightleft=[15,5]
			self.eyesightright=[5,15]		
		elif(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] >= 1):
			self.eyeleftplace[0] = self.X + 5
			self.eyeleftplace[1] = self.Y + 5
			self.eyerightplace[0] = self.X - 15
			self.eyerightplace[1] = self.Y + 5
			self.eyesightleft=[15,5]
			self.eyesightright=[15,5]
		elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] >= 1):
			self.eyeleftplace[0] = self.X 
			self.eyeleftplace[1] = self.Y + 5
			self.eyerightplace[0] = self.X - 15
			self.eyerightplace[1] = self.Y
			self.eyesightleft = [5,15]
			self.eyesightright = [15,5]
		elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] == 0):
			self.eyeleftplace[0] = self.X - 5
			self.eyeleftplace[1] = self.Y + 5
			self.eyerightplace[0] = self.X - 5
			self.eyerightplace[1] = self.Y - 15
			self.eyesightleft = [5,15]
			self.eyesightright = [5,15]
		elif(self.directionvector[self.direction][0] <= -1 and self.directionvector[self.direction][1] <= -1):
			self.eyeleftplace[0] = self.X
			self.eyeleftplace[1] = self.Y -15
			self.eyerightplace[0] = self.X -15
			self.eyerightplace[1] = self.Y 
			self.eyesightleft = [5,15]
			self.eyesightright = [15,5]
		elif(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] <= -1):
			self.eyeleftplace[0] = self.X - 15
			self.eyeleftplace[1] = self.Y - 5
			self.eyerightplace[0] = self.X + 5
			self.eyerightplace[1] = self.Y - 5
			self.eyesightleft = [15,5]
			self.eyesightright = [15,5]
		elif(self.directionvector[self.direction][0] >= 1 and self.directionvector[self.direction][1] <= -1):
			self.eyeleftplace[0] = self.X
			self.eyeleftplace[1] = self.Y - 15
			self.eyerightplace[0] = self.X + 5
			self.eyerightplace[1] = self.Y
			self.eyesightleft = [5,15]
			self.eyesightright = [15,5]

	def saveNN(self):
		print(os.path.join(self.cwd,'brains',self.objectname))
		with open(os.path.join(self.cwd,'brains',self.objectname), 'wb') as output:
			pickle.dump(self.nn, output, pickle.HIGHEST_PROTOCOL)
		print ("%s saved" % (self.objectname))

	def saveViableNN(self):
		with open(os.path.join(self.cwd,'brains',(("%s.viable.pybrain_pkl")%(self.objectname))), 'wb') as output:
			pickle.dump(self.nn, output, pickle.HIGHEST_PROTOCOL)
		print ("%s saved" % (self.objectname))

	def saveEaterNN(self):
		with open(os.path.join(self.cwd,'brains',(("%s.viable_eater.pybrain_pkl")%(self.objectname))), 'wb') as output:
			pickle.dump(self.nn, output, pickle.HIGHEST_PROTOCOL)
		print ("%s saved" % (self.objectname))

	def saveNotViableNN(self):
		with open(os.path.join(self.cwd,'brains',(("%s.pkl_noviable")%(self.objectname))), 'wb') as output:
			pickle.dump(self.nn, output, pickle.HIGHEST_PROTOCOL)
		print ("%s saved" % (self.objectname))

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
				trainingoutputs = self.gettraining([hungri, foodtoeat, gotoleft, gotoright])

				neuraloutputs = self.responce([hungri, foodtoeat, gotoleft,gotoright])

				if(dontPrintInfo == False):
					#print ("IN", [hungri, foodtoeat, gotoleft, gotoright],"Tout:", self.roundfloat(trainingoutputs), "Nout:",self.roundfloat(neuraloutputs), [self.foodamount, self.foodavail, self.foodInLeft, self.foodInRight],self.nn.totalerror)
					print ("IN", [hungri, foodtoeat, gotoleft,gotoright], "Nout:",self.roundfloat(neuraloutputs))

				if(round(neuraloutputs[0], 0) == 1):
					if(self.foodavail >= 0.5):
						if(dontPrintInfo == False):
							print (self.printEpilogue, "eated", self.foodavail,"OK")
						
					else:
						if(dontPrintInfo == False):
							print (self.printEpilogue, "eated", self.foodavail,"NOK")
							print ("IN", [hungri, foodtoeat, gotoleft,gotoright], "Nout:",self.roundfloat(neuraloutputs))
						
					self.foodamount = self.foodamount + self.foodavail - self.consumption #- (neuraloutputs[1]/2)
					self.eaten = "eated"
					#print (self.printEpilogue, "eated", self.foodavail,"OK", self.foodamount)
					#print("soi", round(neuraloutputs[0], 0) , neuraloutputs[0])
				else:
					self.foodamount = self.foodamount - self.consumption #- (neuraloutputs[1]/2)
					self.eaten = "full"
					#print("ei syoty")

				#self.foodamount = self.foodamount - self.consumption + (self.foodavail) #- (neuraloutputs[1]/5)
				
				if(round(neuraloutputs[1],0)==1):
					if(self.directionvector[self.direction][0] == 0 and self.directionvector[self.direction][1] == 0):
						self.randmovevector()
					self.speed = neuraloutputs[1] * 7
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

				if(round(neuraloutputs[2],0) == 1): #move left
					if (self.direction % 2 ==  0):
						if(self.direction == 0):
							self.direction = 6
						else:
							self.direction -= 2
					elif(self.direction == 0):
						self.direction = 7
					else:
						self.direction -= 1
					self.seteyes()
					#if(dontPrintInfo == False):
					#	print ("turn left",self.direction, self.directionvector[self.direction])
				elif(round(neuraloutputs[3],0) == 1): #move right
					if (self.direction % 2 ==  0):
						if(self.direction == 6):
							self.direction = 0 
						else:
							self.direction += 2
					elif(self.direction == 7):
						self.direction = 0 
					else:
						self.direction += 1
					self.seteyes()
					#if(dontPrintInfo == False):
					#	print ("turn right",self.direction, self.directionvector[self.direction])

				if(self.X >= self.eartsize[0]):
					self.X = 0
					self.randmovevector()
					self.seteyes()
				elif(self.X <= 0):
					self.X =self.eartsize[0]
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
					
					if(dontPrintInfo == False):
						print (self.movememory, self.movecount)
				else:
					self.movememory = self.roundfloat(neuraloutputs)
				self.movecount += 1
			except:
				print ("Unexpected error:", sys.exc_info())
				self.eated = ""
				self.move = "exception dood"


		#print (self.roundfloat(trainingoutputs), self.roundfloat(neuraloutputs), self.roundfloat([self.foodamount, self.foodavail]))

	def didoueat(self):
		if("eated" in self.eaten):
			#print(self.objectname, "eaten")
			self.eaten = "full"
			return True
		else:
			return False

	def addfoodavail(self,addfood):
		self.foodavail =  addfood

	def turnleft(self, foodInLeft):
		self.foodInLeft = foodInLeft

	def turnright(self, foodInRight):
		self.foodInRight = foodInRight

	def randmovevector(self):
		vectorok = 1
		loops=0;
		temp = random.randint(0,7) 
		while(vectorok):
			if(temp == self.direction):
				temp = random.randint(0,7)
			else:
				self.direction = temp
				vectorok = 0
				

	def roundfloat(self, rounuppilist):
		roundedlist = []
		for i in range(len(rounuppilist)):
			roundedlist.append(round(rounuppilist[i],0))
		return roundedlist

	def getliveinfo(self):
		time2 = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
		diff = time2 - self.startime
		return [round(self.foodamount,4), round(self.speed,4), self.filename, diff.total_seconds(), self.movecount]

	def areyouallive(self):
		time2 = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
		diff = time2 - self.startime
		if(self.foodamount < -5.00 or self.foodamount > 5.00):
				if (self.foodamount > 5.00):
					self.saveEaterNN()
				else:
					self.saveNotViableNN()
				return (["dood", self.foodamount, self.move])
		if(diff.total_seconds() == 300):
			if(self.foodamount < -10.01 or self.foodamount > 10.00):
				self.saveNotViableNN()
				return (["dood","300s", self.foodamount, self.move])
		elif(diff.total_seconds() == 400):
			if(self.foodamount < -4.01 or self.foodamount > 4.00):
				self.saveNotViableNN()
				return (["dood","600s", self.foodamount, self.move])
		elif(diff.total_seconds() > 600):
			self.saveViableNN()
			#self.saveLog(self.filename, self.nn.inspectTofile(), 'a+')
			return (["icanseemyhousefromhere"])
		if(self.move == "exception dood"):
			return (["dood",self.move])
		return ["ok"]
	
	def returnname(self):
		return self.objectname
