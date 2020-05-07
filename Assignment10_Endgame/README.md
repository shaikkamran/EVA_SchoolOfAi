## Self Driving Car on a City Map using TD3
For All the code for TD3 and explanation with steps please refer `https://github.com/shaikkamran/Reinforcement_learning-T3D-algorithm`


I have done this Project in mainly four steps.
1. Just a Naive integration of the T3D algorithm with the car (kivy environment ) This was already there.
2. Figured out the changes required in the environment .
    
    * Like changing the car's position when called reset_function
    * Getting the cropped image of the mask.png with car in place .All of the code for this is there in car_crop_utils.py
     
    * Then wrote functions like calculate orientation of the car wrt to the goal.For this I realized that reward function is the one where my model is lacking.Reward function must be continuous for the gradient optimization.I could have solved it perfectly by making the car just to move on road by only tweaking the reward function.

3. Had change the TD3 actor and critic models to a CNN model which i have made for a varible image input size.Cnn model is the a small network which could give 99.2 percent acc on mnist in 12 epochs..

4. All the images used for this project can be found in images folder

5.Even after doing this the circling issue persisted.For that I had to include a final fully connected layer in the CNN and also included another parameter which was now an extra parameter called orientation which was getting passed to the Actor model.

6.Doing all these steps on a blank image / map with specific goals helped me solving this issue of car rotating in circles at the same position.[a link]https://www.youtube.com/watch?v=CD-yiaY0uH8&t=55s

7.Now I found the hyperparameters to be used to train the TD3 algorithm .and the task was to train it on the actual map `mask.png`
<br>
8.For this I played with rewards and added rewards for when the car is out of road and other conditions like when it hits the wall.
<br>
10.Finally I was able to train my car to reach the specified goals properly but the car sometimes comes out of the road in order to reach the goals fast (This is what I have to fix.And this can be mostly solved by actually tweaking the rewards ).
[a link]https://www.youtube.com/watch?v=IEt5RREDBnc
<br>
11. Improvements to be done are :-
    * Make the car move on roads to reach the goal whereever it is getting distracted.

12. All the training can be done be running python map.py
13. t3d_small is the main file where all the t3d code is written
14. inferencing can be simply done by doing python infer.py

Requirements python 3.7 with pytorch ,kivy,cv2,numpy,matplotlib  .

Thanks for all the guidance - 
         https://www.linkedin.com/in/rohanshravan/<br>
         https://www.linkedin.com/in/the-school-of-ai-78288b194/<br>



 
    
