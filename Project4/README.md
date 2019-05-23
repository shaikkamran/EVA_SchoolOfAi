## Architectural Basics

**How many layers** -- Depends on the receptive field we want to achieve to solve the problem.
**Image Normalization**-- dividing pixel values by 255 ,bringing everything into a range of [0,1] for treating all the images as in equal resolution.
**MaxPooling** --Max pooling is used to take out the pixel values from the image which are more intensified ,in order to process on the much useful information ignoring the fact that we are losing some meaningful information too.
**1x1 Convolutions**-One 1x1 convolution for transition block (merging the most important features ,without actually performing typical convolution with less parameters,should be placed after Max-ppoling.
**3x3 Convolutions**-- 3x3  convolutions are basic convolutions used everywhere after 2015 .they provide the best convolution information with least no of paramters/weights ,faster on nvidia-gpus and preserves symmetry too.
**Receptive Field**-- receptive field is the area of the image whcih a pixel can see in any layer of Dnn.Lets say the image is 32x32 and most of the objects in images are 10x10 ,then the receptive field of 10x10 will be suffiecient for the network.
**SoftMax**--This is probablity like function (Can we say probability exponentials ).This is basically used as a final layer activation for classification.which penalizes the lesser values to even even lesser.
**Learning Rate**--
**Kernels and how do we decide the number of kernels?**--Depend on the no of  classes we have, as well as how varied each class is. The higher the complexity of the dataset, higher no. of kernels are required to capture those features.
**Batch Normalization**--Batch normalization is the channel normalization in our case,which is basically making every value in the range of [-1,1] ,this is done so as to treat every channel in a similar fashion.
**Position of MaxPooling**--initially the positioning of max pooling must be after we know that our network has sucessfully extracted the gradients and textures from the image.i.e, after three convolutions and the same 3conv -max pooling -3 conv can continue till we reach the receptive field and also this must be a bit far from the prediction layer.
**Concept of Transition Layers** -- Transition layers are used for discarding the less useful kernels and taking away those more important kernels and also reducing the no of parameters of the network ,(1x1 Conv is used). 
**Position of Transition Layer** -- Positioning os transition layers can be after max pooling or when we know that we have too many channels and we need not carry forward each one of them for our use case.
**Number of Epochs and when to increase them**-- No of epochs should be less while experimenting with the architecture types and deciding on various architectures with different hyper parameters and incremental changes.Once all of this is done efficiently we can train the model for as many no of epochs we want until we overfit.
**DropOut**-reduces the gap between training and val acc.Used for avoiding overfitting.randomly makes x no of channels zero hence forcing the others to become better and improve .
**When do we introduce DropOut, or when do we know we have some overfitting**-- I could not figure out any rule to when to introduce dropout(can be used after every conv bn ) ,but we dont usually do this before the prediction and while the transition ,because this can cause effectively learning and then losing kind of scenario.
**The distance of MaxPooling from Prediction**--Max pooling should be like  atleast 3 convolutions far away from prediction beacuse max pooling anything later from this might cause losing of important info and will not let out networ to learn better on the classifying stages. 
**The distance of Batch Normalization from Prediction**-- Can be used after every Conv but not to be used before the prediction layer .
**When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)**--When we have reached the desired receptive field.
**How do we know our network is not going well, comparatively, very early**-- When our first three training accuracies are not well compared to the previous architecture and also when there is a huge difference between val loss and train loss.
**Batch Size, and effects of batch size**--There is no optimal unique value for batch size ,increasing batch size can show improvements in accuracy upto some extent but after that it can also decrease the accuracy.there is no fixed batch size.For each different dataset an optimal batch size may vary.A very large batch size may reduce fine learning of the very minute features but computation is faster ,iterations in each epoch can become smaller .
**When to add validation checks**-After each epoch
**LR schedule and concept behind it**-- Concept is that if we maintain the same lr throught the epochs we might overshoot the local minima.Hence reduceOnplateu and cyclic learning rates are a great option .
**Adam vs SGD**-Adam more like Vanilla gradinet descent,Sgb ,updates after each mini batch.val -acc graph is more haphazard in case of sgd.
**Modelcheckpoints**-saves/retains the best model after every epoch.

