I have implemented crime recognitions from cctv footages using UCF-crime dataset which can be obtained from [here](https://webpages.uncc.edu/cchen62/dataset.html). The dataset being too big I downloaded shorter version of it available on [kaggle](https://www.kaggle.com/mission-ai/crimeucfdataset). This shorter version consists of 8 classes which includes Abuse, arrest, arson, assault, burglary, explosion, fighting and normal. 

# Datapreprocessing 
I converted videos into frames and took only 16 frames from every video for the training of model. These 16 frames were selected from complete video sequence by skipping frames according to video length. In this dataset the number of videos are less but longer so to increase number of samples by 10 times I took 16 samples where first frame started from 0-9 thus giving 10 times the number of videos and all with different images. To speed up the transfer of data I combined these 16 images into 1. The implementation of the preprocessing can be found in videodata.ipynb. The preprocessed data with 16 samples can be found [here](https://drive.google.com/file/d/1vEk82F35yM9wW5qRZYPH6QDH-BLuZ2nE/view?usp=sharing).


# Model used
This task is of action recognition and I tried one of the best model for action recognition [Slow Fast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf) worked best. The implementation of this network in pytorch can be found [here](https://github.com/Guocode/SlowFast-Networks). 

# Training
The model trained fast and reached a training accuracy of 99% within 20 epochs.
![](https://github.com/sanchit2843/Videoclassification/blob/master/assets/acc.png)

![](https://github.com/sanchit2843/Videoclassification/blob/master/assets/loss.png)
