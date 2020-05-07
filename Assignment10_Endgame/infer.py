# Self Driving Car

import pickle
import time
from random import randint, random,uniform
from collections import namedtuple
import cv2
import imutils
import matplotlib.pyplot as plt
# Importing the libraries
import numpy as np
import torch
# Importing the Kivy packages
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color,  Line
from car_crop_utils import get_car_paste_cordinates,get_half_cords,get_sand_crop_coordinates,car_on_corners

from kivy.properties import (NumericProperty, ObjectProperty,
                             ReferenceListProperty)
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.vector import Vector
from PIL import Image as PILImage
import logging
from logger import get_logger

logger = get_logger("./logs")
logging.Logger.manager.root = logger

from td3_small import TD3, ReplayBuffer,State

action_dim=1
orientation_dim=1
out_channels=10
critic_out_channels=10
max_action=2
state_size=(3,50,50)
image_crop_size=200
policy = TD3(state_size,action_dim, max_action,orientation_dim,out_channels,critic_out_channels)
policy.load("car_t3d247037","models_5_5_2020")

replay_buffer=ReplayBuffer()

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0


action2rotation=np.random.uniform(-max_action,max_action,100)
reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
state=namedtuple('state',['image','orientation'])

sand_img=PILImage.open("./images/mask.png")
# with open("replay.pkl","rb") as file:
#     replay_buffer.storage=pickle.load(file)

# textureMask = CoreImage(source="./kivytest/simplemask1.png")
counter=0
car=PILImage.open("./images/arrow_resized.png")
sand_img=PILImage.open("./images/mask_with_border.jpg")
# Initializing the map
first_update = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sand_coordinates=np.where(cv2.imread("./images/mask.png")==0)


seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 


total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()
episode_timesteps=0
max_episode_steps=3000

def init():

    global sand
    global goal_x
    global goal_y
    global first_update
    
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    
    # goal_x = 1420
    # goal_y = 622
    goal_x = 1051
    goal_y = 594
    
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0
episode_reward=0
# Creating the car class


def convert_to_tensor(obs,no=image_crop_size//2):
    return torch.tensor(obs,dtype=torch.float).expand(1,-1,-1,-1).to(device)

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def reset_env(self):
        global sand_coordinates
        no=randint(20000,len(sand_coordinates[0]))
        
        # logger.info(f"got={sand_coordinates[0][no]}{sand_coordinates[1][no]}")
        self.x=int(sand_coordinates[0][no])
        self.y=int(sand_coordinates[1][no])
        
        self.angle=uniform(-360,360)

    def move(self, rotation):

        # logger.info(f"rotation is {rotation}")
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        # logger.info(f"self.angle={self.angle}    self.rotation={self.rotation}")
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.
        

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(2, 2)

    def get_orientation(self):

        global goal_x,goal_y

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        
        orientation = Vector(*self.car.velocity).angle((xx,yy))
        
        # return [orientation,-orientation]
        return [orientation]

    def get_state(self,no=image_crop_size//2):

        x,y=int(self.car.x),int(self.car.y)

        car_rotated=car.rotate(self.car.angle,expand=True)

        logger.info(f"car position={x} {y}")
        
        car_final=car_rotated.crop(car_on_corners(car_rotated,y,x))

        sand1=sand_img.copy()

        sand1.paste(car_final,get_car_paste_cordinates(car_rotated,y,x),car_final)

        sand_crop=sand1.crop(get_sand_crop_coordinates(y,x,no))
        sand_crop_=np.array(sand_crop.rotate(90))/255.0
        sand_crop=sand_crop.convert(mode="RGB")
        s=np.array(sand_crop)/255.0
        cv2.imshow("img",sand_crop_)
        cv2.waitKey(1)
        
        s=cv2.resize(s,(state_size[1],state_size[2]))
        return state(s.reshape(state_size),self.get_orientation())
    
    def calculate_reward(self,last_distance,last_reward,swap,last_orientation):
        
        global goal_x,goal_y
        boundary_no=5
        done=False
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        orientation=self.get_orientation()

        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        # reward_copy=last_reward
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            print(f"orientation{orientation}")
            last_reward = -0.6
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = 0.1
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            print(f"orientation{orientation}")

        if distance < last_distance:
            last_reward += 0.5
        
        else:
            last_reward = last_reward +(-0.2)

        if self.car.x < boundary_no:
            self.car.x = boundary_no
            last_reward += -1
            done=True
        if self.car.x > self.width - boundary_no:
            self.car.x = self.width - boundary_no
            last_reward += -1
            done=True
        if self.car.y < boundary_no:
            self.car.y = boundary_no
            last_reward += -1
            done=True
        if self.car.y > self.height - boundary_no:
            self.car.y = self.height - boundary_no
            last_reward += -1
            done=True
        if distance < 25:
            if swap == 1:
                goal_x = 1051
                goal_y = 594

                swap = 0
                done=True
            else:
                goal_x = 143
                goal_y = 277
                swap = 1

        # if last_reward<=reward_copy and reward_copy<=-0.9:
        #     last_reward=last_reward+reward_copy

        return distance,last_reward,swap,done

    def chose_random_action(self):
        return action2rotation[randint(0,len(action2rotation)-1)]


    def update(self, dt):

        
        global reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap,episode_reward
        global counter,done,episode_timesteps,current_state
        global total_timesteps,max_action,max_episode_steps,episode_num,timesteps_since_eval,policy_freq,policy_noise,tau,discount,replay_buffer
        
        longueur = self.width
        largeur = self.height
        if first_update:
            init()
        
        current_state = self.get_state()

        if done:

            _ = self.car.reset_env()
            current_state = self.get_state()

            done = False

            action = self.chose_random_action()

        print("Action from agent")
        o = np.array([current_state.orientation]).reshape(1, len(current_state.orientation))
        action = policy.select_action(
            convert_to_tensor(current_state.image), torch.tensor(o, dtype=torch.float).to(device)
        )

        if isinstance(action, np.float64):

            self.car.move(float(action))
        else:
            action = action[0]
            self.car.move(float(action))
        distance, reward, swap, done = self.calculate_reward(
            last_distance, reward, swap, current_state.orientation
        )

        last_distance = distance

        





        
# Adding the painting tools





class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        
        Clock.schedule_interval(parent.update, 1.0/60.0)
        
        return parent

    
        

if __name__ == '__main__':

    CarApp().run()
