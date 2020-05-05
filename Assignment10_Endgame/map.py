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
orientation_dim=2
out_channels=1
critic_out_channels=1
max_action=2
state_size=(3,50,50)
image_crop_size=200
policy = TD3(state_size,action_dim, max_action,orientation_dim,out_channels,critic_out_channels)
# policy.load("car_t3d26107","pytorch_models")

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

# action2rotation = [-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0]
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
sand_img=PILImage.open("./images/mask.png")
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
batch_size = 300 # Size of the batch
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


        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.0
        
        return [orientation,-orientation]

    def get_state(self,no=image_crop_size//2):

        x,y=int(self.car.x),int(self.car.y)

        car_rotated=car.rotate(self.car.angle,expand=True)

        print(f"car position={x} {y}")
        
        car_final=car_rotated.crop(car_on_corners(car_rotated,y,x))

        sand1=sand_img.copy()

        sand1.paste(car_final,get_car_paste_cordinates(car_rotated,y,x),car_final)

        sand_crop=sand1.crop(get_sand_crop_coordinates(y,x,no))

        sand_crop=sand_crop.convert(mode="RGB")
        s=np.array(sand_crop)/255.0
        # cv2.imshow("img",s)
        # cv2.waitKey(1)
        # print(s.shape)
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
            last_reward = -0.5
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = 0.2
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            print(f"orientation{orientation}")
        if distance < last_distance:
            last_reward += 0.8

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
        
        if counter==0:
            
            current_state=self.get_state()
            plt.imsave("car_image.png",current_state.image.reshape(50,50,3))
            counter+=1
        
       
# We start the main loop over 500,000 timesteps
        if total_timesteps < max_timesteps:
        
        # If the episode is done
            if done :

                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                # if total_timesteps>=10000:
                    logger.info("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    # policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                    policy.train(replay_buffer,min(episode_timesteps,100), batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    logger.info("Saving the model")
                    timesteps_since_eval %= eval_freq
                    # evaluations.append(evaluate_policy(policy))
                    file_name="car_t3d"+str(total_timesteps)
                    # replay_file="car_t3d_replay"+str(total_timesteps)+".pkl"
                    policy.save(file_name, directory="./pytorch_models")
                    # with open(f"./replay_buffers/{replay_file}","wb") as file:
                    #     logger.info("Saving the replay buffer")
                    #     pickle.dump(replay_buffer.storage,file)
                    # # np.save("./results/%s" % (file_name), evaluations)
                    
                # When the training step is done, we reset the state of the environment
                _=self.car.reset_env()
                current_state= self.get_state()
                
                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
            
        # Before 10000 timesteps, we play random actions
        if total_timesteps < start_timesteps:
            # action = env.action_space.sample()
            # action=action2rotation[randint(0,2)]
            logger.info("Random Action from environment")
            action=self.chose_random_action()
        
        else: # After 10000 timesteps, we switch to the model
            # action = policy.select_action(np.array(obs))
            logger.info("Action from agent")
            o=np.array([current_state.orientation]).reshape(1,2)
            action=policy.select_action(convert_to_tensor(current_state.image),torch.tensor(o,dtype=torch.float).to(device))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=1)).clip(-max_action, max_action)
        
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        # new_obs, reward, done, _ = env.step(action)
        if isinstance(action,np.float64):

            self.car.move(float(action))
        else:
            action=action[0]
            self.car.move(float(action))
        distance,reward,swap,done=self.calculate_reward(last_distance,reward,swap,current_state.orientation)
        last_distance = distance

        logger.info(f"Episode Timesteps={episode_timesteps}\nTotal Iterations={total_timesteps}\ndistance={distance}\nlast_reward={reward}\nswap={swap}\norientation{current_state.orientation}\ngoal_x,goal_y={goal_x} {goal_y}\nrotation={self.car.rotation}\nvelocity={self.car.velocity}\nangle={self.car.angle}")
        
        next_state=self.get_state()


        # We check if the episode is done
        done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
        if episode_timesteps==max_episode_steps:
            done=True
        # if total_timesteps%1000==0:
        #     done=True
        # We increase the total reward
        episode_reward += reward
        
        
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
            
        replay_buffer.add((current_state,next_state, action, reward, done_bool))
        
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        current_state = next_state
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1







        
# Adding the painting tools






class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    
    
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        # sand = np.zeros((longueur,largeur))

    def save(self, obj):
        logger.info("saving brain...")
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        logger.info("loading last saved brain...")
        

# Running the whole thing

if __name__ == '__main__':

    CarApp().run()


def on_exit():
    logger.info("Saving the replay buffer")
    with open("replay.pkl","wb") as file:
        pickle.dump(replay_buffer.storage,file)

# on_exit()