## Self Driving Car on a City Map using TD3
For All the code for TD3 and explanation with steps please refer `https://github.com/shaikkamran/Reinforcement_learning-T3D-algorithm`


I have done this Project in mainly four steps.
1. Just a Naive integration of the T3D algorithm with the car (kivy environment ) This was already there.
2. Figured out the changes required in the environment .
    
    * Like changing the car's position when called reset_function
    * Getting the cropped image of the mask.png with car in place .For this I have done superimposing of images like we did in gradcam.
      As in Rotate car image to the particular angle to which it was in that exact state.and pad it with the extra pixels that
      are required to make the cropped mask image and this rotated car image equal and then 
      superimpose them by ```<i>cv2.addWeighted(car_state_in3d,0.35,car_oriented_padded,0.55,0)</i>```
    
    * Then wrote functions like calculate orientation of the car wrt to the goal.

3. Had change the TD3 actor and critic models to a CNN model which i have made for a varible image input size.Had to figure out gap and padding part in pytorch as its not pretty straight forward like keras/tensorflow.Code for the same can be found in t3d.py.

4. All the images used for this project can be found in images folder
5. After doing all these I realized that my car is actually not learning anything after its trained it starts rotating in circles.Figured out that i was giving a very small image with which its not looking at the roads at all in most of the scenarios .Then I cropped a 100* 100 image which i resized to 50 * 50 and sent to the network.Also instead of car i placed a pink triangle with its orientation similar to that of car in order to avoid My CNN model to overwork on learning complex features of the car.
6.Even after doing this the circling issue persisted.For that I had to include a final fully connected layer in the CNN and also included another parameter which was now an extra parameter called orientation which was getting passed to the Actor model.

7.Doing all these steps on a blank image / map with specific goals helped me solving this issue of car rotating in circles at the same position.[a link]https://www.youtube.com/watch?v=CD-yiaY0uH8&t=55s

8.Now I found the hyperparameters to be used to train the TD3 algorithm .and the task was to train it on the actual map `mask.png`
<br>
9.For this I played with rewards and added rewards for when the car is out of road and other conditions like when it hits the wall.
<br>
10.Finally I was able to train my car to reach the specified goals properly but the car sometimes comes out of the road in order to reach the goals fast (This is what I have to fix.And this can be mostly solved by actually tweaking the rewards ).
[a link]https://www.youtube.com/watch?v=SNuIU1w3CmQ
<br>
11. Improvements to be done are :-
    * Make the car move on roads to reach the goal whereever it is getting distracted.

12. All the training can be done be running python map.py
13. inferencing can be simply done by doing python inference.py

Requirements python 3.7 with pytorch ,kivy,cv2,numpy,matplotlib  installed.

Thanks for all the guidance - 
         https://www.linkedin.com/in/rohanshravan/<br>
         https://www.linkedin.com/in/the-school-of-ai-78288b194/<br>



 
    
