## Self Driving Car on a City Map using TD3
For All the code for TD3 and explanation with steps please refer `https://github.com/shaikkamran/Reinforcement_learning-T3D-algorithm`<br>

**Aim** To make our agent/car travel from one goal to another from roads.
![alt text](https://github.com/shaikkamran/EVA_SchoolOfAi/blob/master/Assignment10_Endgame/images/citymap_with_goals.png)


**Backgroung Image** The actual image used for training is the hand drawn mask of the above image.
![alt text](https://github.com/shaikkamran/EVA_SchoolOfAi/blob/master/Assignment10_Endgame/images/MASK1.png)

- **Environment**-Created by kivy `map.py`
- **Algorithm**  -Td3 algorithm used for training ```t3d_small.py```

Any Reinforcement learning project consists of environment ,Reward function ,Episode,Action of the agent and its task.
In our case 
- **Actions are** - Car's velocity and rotation
- **Reward function** - A continuous reward function which is like = `(1-distance)^0.4`
    The reward function is built by taking several other parameters into consideration such as velocity, episode_timesteps etc.
- **Episode** -In the project one episode gets completed when the car reachess both the goals (positive terminals).Or if it hits the boundaries (walls) negative terminals.
Steps taken.
1. Creation of a Kivy environment.<br>
    Main functions involved are<br> 
    **get_state and car_crop_utils** (get the current cropped image of the portion where the car is)<br>
    For cropping PIL library is used.
    **move** helps car in moving with the values provided (rotation,velocity)<br>
    **calculate_reward** All the rewarding logic can be found here<br>
    **reset_env**-reinitialize car at any random point in point in the map.<br>

2. Integration of the T3D algorithm .
   For both the Actor and Critic Bunch of Conv layers + linear layers are written.
   Idea is the Image (Convolutions) will help our car to stay on the road and other state parameters such as distance and orientation of car towards the goal (Linear) layers will help the car decide the direction  
   Cnn model is the a small network which could give 99.2 percent acc on mnist in 12 epochs..

3. All the images used for this project can be found in images folder

4.Challenges faced-The rewarding function should not be a step function.And negative rewards must not be very out of order .
I was facing car rotation issue where the car goes on rotating in at the same place .For that i am giving a rotattion penalty and another issue I faced was car hitting the negative terminals.This also was solved by adjusting and tweakig the reward function.(Basically what you teach is what it learns)

5.Doing all these steps on a blank image / map with specific goals helped me solving this issue of car rotating in circles at the same position.[a link]https://www.youtube.com/watch?v=CD-yiaY0uH8&t=55s

6.Now I found the hyperparameters to be used to train the TD3 algorithm .and the task was to train it on the actual map `mask.png`
<br>
7.For this I played with rewards and added rewards for when the car is out of road and other conditions like when it hits the wall.
<br>
8.Finally I was able to train my car to reach the specified goals properly but the car sometimes comes out of the road in order to reach the goals fast (This is what I have to fix.And this can be mostly solved by actually tweaking the rewards ).
[a link]https://www.youtube.com/watch?v=IEt5RREDBnc
<br>

9. All the training can be done be running ```python map.py```
10. t3d_small.py is the main file where all the t3d code is written.
11. inferencing can be simply done by doing ```python infer.py```

Requirements python 3.7 with pytorch ,kivy,cv2,numpy,matplotlib  .

Thanks for all the guidance - 
         https://www.linkedin.com/in/rohanshravan/<br>
         https://www.linkedin.com/in/the-school-of-ai-78288b194/<br>



 
    
