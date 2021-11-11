# Objective: ASL Recognition using Convolutional Neural Network


## Motivation/Background:

American Sign Language (ASL) is one of the most commonly used sign languages in the world. By some estimates, ASL is used by about [250,000 to 500,000](https://www.gallaudet.edu/documents/Research-Support-and-International-Affairs/ASL_Users.pdf) people in North America alone. Various statistics show that those with hearing disabilities are highly disadvantaged as compared to their peers.

|Category|Grouping|Normal|Disabled|Difference|
|--------|--------|------|--------|----------|
|Education|High School Dropout Rate|18.7%|44.4%|2.37x|
||College Graduation Rate|12.8%|5.8%|0.45x|
||Post College Graduation Rate|9.2%|4.8%|0.52x|
|Employment|16+ yo|91.2%|83.9%|0.92x|
||16 - 44 yo|82%|58%|0.71x|
||45 - 64 yo|73%|46%|0.63x|
|Income|$10K - $25K|26%|28%|1.08x|
||$50K+|29%|14%|0.48x|

With these statistics in mind, we should ask ourselves: beyond their physical disability, how much **more** should it cost to be disabled?

That begs the question: how can we facilitate the inclusion of deaf people into society?

With the rise in remote work and increased use of video conferencing tools such as Google Meet, Zoom and Microsoft Teams, one way we can help facilitate their participation is by embedding a sign language interpreter for the deaf to speak to their non-deaf peers. YouTube currently has a tool to create text captions based on sound and language recognition. Building upon that idea, this project aims to create a tool to recognize ASL sign languages to be interpreted by text captioning.

However, understanding my current limits, my current goal is to train and recognize static alphabetical gestures in ASL.


# Methodology

## Libraries Used:

1. OpenCV2
2. Tensorflow, Keras
3. VGG16
4. Regex
5. Shutil
6. Numpy
7. Matplotlib
8. Pandas
9. OS

## Datasets

|No.|Folder name|Description|
|---|-----------|-----------|
|1.|Gestures|Dataset of 26 gestures split into training and testing set for model training|
|2.|Models|Various models trained with different hyperparameters|
|3.|Results|Dataset of results after model is tested onto the testing set (CSV)|
|4.|History|Model metrics (accuracy, loss, val_accuracy, val_loss)|
|5.|CreateGestures.py|Python file to create dataset.|

Some of the datasets were too large to be pushed onto GitHub. The links to these datasets are below:

Models: https://drive.google.com/drive/folders/1TkE0VFxvOM9LvV12UGxh1qNKA5R-lHKp?usp=sharing

Gestures dataset: https://drive.google.com/drive/folders/1jfvGZc1hsvgfKYH07Kwe08qsqCrdnfpW?usp=sharing


## Project Flow


### 1. Create Dataset

Credits to [Akshay Bahadur](https://github.com/akshaybahadur21) for general function to create dataset.

**General method:** 
I used OpenCV to create my dataset. The threshold was set such that the images taken were only limited to my hand (the region of interest) so that it does not take into account my whole screen as it was unnecessary to the ASL gestures in this project. 

**Number of images taken:**
Images per gestures: 2,000
5 Alphabets (Gestures_5): 10,000 (A - E)
Total images (Gestures): 52,000 (26 alphabets)

**Image size:**
224 x 224. Image size were set to fit with VGG16 model.

**On gestures:** 
While ASL has predetermined gestures of alphabets, there were two things which came to mind: some of the suggested gestures for the same alphabets were different. Therefore, I created a dataset based on my interpretation of the gestures. Furthermore, since this is a basic alphabet gesture recognition, I had used a placeholder for alphabets that were dynamic, i.e. alphabets that are interpreted with movement (J and Z). J was replaced with the [Vulcan Salute](https://cdn.shopify.com/s/files/1/1061/1924/products/Vulcan_Salute_Emoji_Icon_ios10_grande.png?v=1571606113) while Z was replaced with a [five](https://hotemoji.com/images/dl/c/raised-hand-with-fingers-splayed-emoji-by-google.png).

**On morphology:** 
Since some of the gestures are very similar, some morphologies work better than others. For instance, using a white-filled morphology (```MORPH_OPEN``` or ```MORPH_CLOSE```) would cause gestures with folded fingers to look similar (e.g. A vs E). The best morphology should provide the best distinction between the gestures. As such, I chose ```MORPH_GRADIENT``` which takes the outlines of the gestures including the outlines of fingers, folded or otherwise.

**Tried and failed list:**
1. Using background filter: It seemed like a good idea to use background filtering to filter out the background from my region of interest. So I tried using ```cv2.createBackgroundSubtractorMOG2()``` and ```fbag = cv2.createBackgroundSubtractorKNN()```. While it did take out my background, it eventually took my hand as a background too, and so, this method failed.
2. Using other morphologies: As explained above, other morphologies did not work well because it either resulted in images that (i) were too similar or (ii) identified too little (```MORPH_TOPHAT```).

### 2. Prepare Dataset

**Overall idea:**
1. Split dataset into a training set and a testing set, where the testing set is a blind dataset for the trained model to be evaluated upon after optimization.
2. Using ImageDataGenerator, I split the training set into 'train' and 'validation' set to train the model.


### 3. Presets and Functions

**Overall:**
This section contain the model presets and functions that I used to train the model. These include:

|Name|Remarks|
|----|-------|
|Presets|Presets used in the model for setting steps per epoch and batch size.
|Model|Function to instantiate and fit model|
|Checkpoints|Function for callbacks|
|History|Function to save model history metrics|
|Plot|Function to plot history metrics (accuracy, val_accuracy, loss, val_loss)|



### 4. Train Model

**Overall:**
I used two pre-trained models, VGG16 and ResNet50 with ```include_top=False``` and added a few layers and dropout rates. I found that while ResNet50 had many more layers, the VGG16 model with a few added layers had performed better. The optimized 26-gesture model will more thoroughly discussed in the next section.

**On optimization:**
Optimization was done with trial-and-error while keeping an eye on the accuracy and val_accuracy. Then, I evaluated the model on the test set (blind test set).

**On freezing and adding Layers:**
Freezing and adding layers onto the models played a significant impact on model training. In general, more layers helped prediction rates.

**On dropouts:**
Dropouts are used to regularize the model to prevent overfitting. Using dropouts result in [multiple independent internal representations being learned](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/) by the network and thus prevents the model to be too reliant on any specific neuron. In optimizing the model, I used various dropout rates and evaluated the results based on the different tweaks.

**On batch size:**
A higher batch size allows more accurate results, but takes a much longer time. I had initially set batch size to 32, which did not turn out well and my val_accuracy to consistently become 3.846%, which is equal to 1/26. This is equal to the alphabets version of flipping a coin. Since I had 26 gestures, a batch size of 32 was too small for the model to learn and meaningful pattern. Therefore, I changed it to 256, which worked much better since it meant that on average, the model trained about 10 of each class in each batch.

**On steps per epoch:**
A higher number of steps per epoch would increase model accuracy. However, due to computational and time limitations, I decided to use ```total number of obervations / batch size``` amount of steps. 

**On epochs**
After several iterations, I found that it did not really matter how many epochs I had set because I had set my checkpoints to include ```EarlyStopping```. In general, setting ```epochs = 50``` was enough to allow ```EarlyStopping``` to kick in.

**On callbacks**
Three callback functions were used: ```EarlyStopping```, ```ReduceLROnPlateau``` and ```ModelCheckpoint```.
1. Early stopping causes the model to stop when the ```patience``` condition was met. I set ```patience = 7```, which means that if the model did not improve after 7 epochs, the model will stop learning.
2. ReduceLROnPlateau reduces the learning rate when there is no improvement when the ```patience``` condition was met. I set ```patience = 3```  and ```factor = 0.6```, which means that the model will reduce its learning rate to 0.6x when the model does not improve after 3 epochs. I had set the ```patience - 7``` to allow learning rate to reduce twice and then train on one more epoch before stopping. Visually, it's as such:
> Early stopping patience (7) = 2 x Learning rate reduction (2 x 3) + 1 more epoch (1)
3. ModelCheckpoint is used to set our optimization criteria. Since my aim is to best predict an input image, I set ```monitor = val_accuracy```.

### 5. Test Model

**Overall:**
The trained model is tested onto the test set (blind test) and evaluated accordingly.

**Evaluating model:**
Using the ```model.evaluate``` function, the model resulted in an accuracy of over 97%. This result seemed pretty suspicious to me, and so I evaluated it in another way.

**Create function to evaluate model (Results folder):**
I created a function to evaluate whether or not an image in the dataset was predicted correctly. The results were aggregated into a dataframe and saved as a CSV file with this format:

|Actual Alphabet|Model Prediction|Prediction Probability|
|---------------|----------------|----------------------|
|<center>A</center>|<center>A</center>|1.000|


### 6. Observations/Misclassification Analysis

Evaluating the results, I found a different result. My initial thought was that the model with the highest accuracy and val_accuracy score would do best. This was untrue. I found that some of my previous model versions had performed better. When tested in this format, my performing model obtained a 70.3% accuracy rate as compared to the ```model.evaluate``` result of 97.4% accuracy rate.

This was pretty confusing as the evaluation did not result in the same accuracy rate, so I decided to view some of the data in the test set while also using the model to predict the images. In doing so, I felt more confident that my results function was a better representation of the overall test set results with an accuracy of 70%.

# Conclusions and Recommendations


## Conclusions

Some conclusions from the models.

Epoch size should be adjusted to the number of classes available in the dataset. As mentioned before, when I had set epoch size to 32, there was an average of 1 gesture included per batch, which caused the model to be unable to predict anything substantial. Therefore, I had increased the batch size to 256, which means that on average, there will be about 10 gestures per batch. This provided the model with greater predictive power.

On the number of layers, I found that it was generally unnecessary to train the pre-trained models. As such, I had frozen the pre-trained models. Excluding the top, while including a few added layers of my own, had provided the model with greater predictive power. However, with the added layers, there was a need to adjust dropout size accordingly. A dropout too low or too high would cause the model to overfit or underfit respectively.

On the model results, I found that in general, gestures that were different from others was predicted better. In creating my dataset, I had minded to prevent overfitting to the same gesture, and as such, I had rotated and shifted my hand around to provide it with some noise. As a result, somes gestures looked more similar to other gestures beyond it's original shape. However, this was a necessary trade-off because choosing not to provide some noise into the dataset would cause the models to be too strict in predicting the gestures, and thus, minute differences in gestures would result in a worse accuracy rate.

Overall, this project was pretty fun for me as I explored new libraries (OpenCV, etc), which allowed me to expand my personal toolkit.

## Recommendations

In improving this model, there are a few things that I have in mind:      

1. Increase the dataset size.
2. Increase layers and optimize the models accordingly.
3. Reduce learning rate. The model could be trained even better with more data and minimizing learning rate, which would have taken too much time and computational power to do for a personal project.
3. Implement a live-test whereby a camera can be used to test the model in real time.













