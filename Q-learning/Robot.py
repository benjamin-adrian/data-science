__author__ = 'ben'

import urllib.request
import urllib.parse
import time

real_robot = False;


def state(robot = real_robot) :
	url = "http://localhost:17327/poll"
	#if(robot):
		#with urllib.request.urlopen(url) as f:
		#print(f.read().decode('utf-8'))
		
def up(robot = real_robot) :
	#print("up")
	url = "http://localhost:17327/moveSteps/motor%20A/forward/760/150"
	if(robot):
		with urllib.request.urlopen(url) as f:
			print(f.read().decode('utf-8'))
		state()
		
def down(robot = real_robot) : 
	#print("down")   
	url = "http://localhost:17327/moveSteps/motor%20A/backward/800/150"
	if(robot):
		with urllib.request.urlopen(url) as f:
			print(f.read().decode('utf-8'))
		state()
		
def turn_left(robot = real_robot) :	
	url = "http://localhost:17327/turnSteps/motor%20A/left/400/30"
	if(robot):
		with urllib.request.urlopen(url) as f:
		#print(f.read().decode('utf-8'))
			state()
		
def turn_right(robot = real_robot) :	
	url = "http://localhost:17327/turnSteps/motor%20A/right/400/30"
	if(robot):
		with urllib.request.urlopen(url) as f:
			print(f.read().decode('utf-8'))
		state()

def left(robot = real_robot) :
	#print("left")
	turn_left()
	#time.sleep(0.5)
	up()
	turn_right()
		
def right(robot = real_robot) : 
#	print("right")
	turn_right()
#	time.sleep(0.5)
	up()
	turn_left()