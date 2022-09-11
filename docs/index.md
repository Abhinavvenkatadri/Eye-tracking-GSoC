---
layout: default
---

<img src="https://user-images.githubusercontent.com/52126773/189515274-fe10d739-8972-4972-84ec-79e5cbdd5aa8.png" alt="">

# Gazetrack

This provides the complete guide to trying to improve the current eye trackers by implementing and tuning the previous year’s implementation of the eye trackers.We follow Google’s implementation of the Paper titled “ Accelerating eye movement research via accurate and affordable smartphone eye tracking” and going into much depth of their implementation

## Introduction

It has varieties of applications ranging across usability and user experience research, gaming, driving, and gaze-based interaction for accessibility to healthcare. Smartphone gaze could also provide a digital phenotype for screening or monitoring health conditions such as [autism spectrum disorder](https://jamanetwork.com/journals/jamapsychiatry/article-abstract/206705), [dyslexia](https://www.sciencedirect.com/science/article/abs/pii/0042698994902097?via%3Dihub#!), [concussion](https://www.karger.com/article/abstract/358786) and more. This could enable timely and early interventions, especially for countries with limited access to healthcare services. People with conditions such as ALS, locked-in syndrome and stroke have impaired speech and motor ability. The smartphone gaze could provide a powerful way to make daily tasks easier with the use of gaze for interaction, as recently demonstrated by Google [Look to Speak](https://blog.google/outreach-initiatives/accessibility/look-to-speak/).

## The Dataset

All trained models provided in this project are trained on some subset of the massive [MIT GazeCapture dataset](https://gazecapture.csail.mit.edu/index.php) that was released in 2016. You can access the dataset by registering on the website.The details regarding the values they provide are mentioned in their [github](https://github.com/CSAILVision/GazeCapture).They have images along with their respective json files where they have mentioned values like bounding box coordinates of Eye,Face then informations like total number of frames ,number of face detections,eye detections etc.

### Splits

Before explaining splits
All frames that make it to the final dataset contains only those frames that have a valid face detection along with valid eye detections. If any one of the 3 detections are not present, the frame is discarded.

Hence our dataset is obtained after applying the following filters

1.  Only Phone Data
2.  Only portrait orientation
3.  Valid face detections
4.  Valid eye detections

There are two types of splits that are considered

1.MIT Split

2.Google Split



#### MIT Split

The MIT Split maintains the train test validation split at a per participant level, same as what GazeCapture does. What this means is that a data from one participant does not appear in more than one of the train/test/val sets. The fact that the same person is not there in all the splits , helps the model to train and generalize well.

Overall after the following conditions have been met, the details regarding frames are as follows:

| **Total Frames** | **Number of Participants** | **Train/Validation/Test** |
|------------------|----------------------------|---------------------------|
| 427,092          | 1,075                      | Train                     |
| 19,102           | 45                         | Validation                |
| 55,541           | 121                        | Test                      |


Two models have been included for this split.

1.[Current Implemented Model(MIT Split)](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/Checkpoints/CurrentModel_MITSplit.ckpt)

2.[Previous Implemented Model(MIT Split)](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/Checkpoints/Previous_Implemented_Model_MITSplit.ckpt)

The changes in the model will be explained in the further sections


#### Google Split 

Google split their dataset according to the unique ground truth points. What this means is that frames from each participant are present in the train test and validation sets. To ensure no data leaks though, frames related to a particular ground truth point do not appear in more than one set. The split is also a random 70/10/15 train/val/test split compared to a 13 point calibration split.

Two models have been included for this split.

1.[Current Implemented Model(Google Split)](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/Checkpoints/CurrentModel_GoogleSplit.ckpt)

2.[Previous Implemented Model(Google Split)](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/Checkpoints/Previous_Implemented_Model_GoogleSplit.ckpt)

Overall after the following conditions have been met, the details regarding frames are as follows:

| **Total Frames** | **Number of Participants** | **Train/Validation/Test** |
|------------------|----------------------------|---------------------------|
| 366,940          | 1,241                      | Train                     |
| 50,946           | 1,219                      | Validation                |
| 83,849           | 1,233                      | Test                      |

## The Idea

The plan was to improve last year’s model by going carefully and in further depth through the details mentioned by google in their [supplementary](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-18360-5/MediaObjects/41467_2020_18360_MOESM1_ESM.pdf) and try experimenting with the change of hyperparameters. After comparing it with [last year’s implementation](https://dssr2.github.io/gaze-track/) which was implemented in pytorch. 

There were two changes that were made to the previous model after going through google's implementation.

1. Epsilon Value- The default value of epsilon in Tensorflow is 0.001.The previous year’s model was trained on Pytorch using its default value which is epsilon = 10^-5.This was one of the changes that was made to the model

2. Learning rate schedule params - Google used tf.keras.optimizers.schedules.ExponentialDecay with parameters as:

*   initial learning rate: 0.016
*   decay steps: 8000
*   decay rate: 0.64
*   decay type: ’staircase’

Previous implementation of optimizer was modified to  StepLR with parameters step_size as 8000 and gamma as 0.64

## Results 

### Comparison of Model

After comparing last year's google split model with the updated implementation

| **Dataset Name**                        | **Number of Files** | **Current Model** | **Previous Implemented Model** |
|-----------------------------------------|---------------------|-------------------|--------------------------------|
| Google Split, All Phones; Only Portrait | 83,849              | 1.677cm           | 1.86cm                         |

This model was trained on 100 epochs with batch size as 256.
Few Outputs are shown below where the comparison is being done between last year’s model and the updated model




### SVR Implementation

The next work was on improving the SVR results.Google uses  the personalized gaze estimation model which consists of a multilayer feed-forward convolutional neural network (CNN) model 1,2 .Additionaly the output of the penultimate layer(1,4) is extracted and is fitted at a user-level to build a high-accuracy personalized model.This improves the accuracy of the model.

Once the output of the penultimate layer is obtained(1,4) an SVR is applied.For this we have used the test data of the trained model.For obtaining the (1,4) value of the penultimate layer hook is applied to the model.We compare the results of both last year’s and the updated model.

We select 10 users from the test set based on the number of frames and the results are provided on that.The test set is used for fitting the SVR as this is the data the model is not trained on.While fitting we also consider 30 unique point(Ground truth) .The results for both are provided in the table below. 

For sweeping the parameters we consider:

*   kernel=’rbf'
*   C=20
*   gamma=0.6

The Multiouput regressor's epsilon valui was sweeped between 0.1 and 100 to find the optimum value.For fitting the SVR the set is first randomly divided into 70:30 split.We then consider 3 fold cv while applying the grid search.Using this once the best parameter is obtained the results are obtained on the 30% of the data.

The below results are obtained using  this year’s model on the MIT split as mentioned in the previous section:

| User ID | Number of Frames | MED(MIT Split) | After SVR(3 fold)(Considering all frames) | After SVR(3 fold)(Considering 30 unique points) |
|---------|------------------|----------------|-------------------------------------------|-------------------------------------------------|
| 3183    | 874              | 1.86cm         | 1.30cm                                    | 1.15cm                                          |
| 1877    | 860              | 2.09cm         | 1.23cm                                    | 1.19cm                                          |
| 1326    | 784              | 1.78cm         | 1.36cm                                    | 2.02cm                                          |
| 3140    | 783              | 1.71cm         | 1.68cm                                    | 2.47cm                                          |
| 2091    | 788              | 1.86cm         | 1.77cm                                    | 1.87cm                                          |
| 2301    | 864              | 1.69cm         | 1.08cm                                    | 0.99cm                                          |
| 2240    | 801              | 1.69cm         | 1.26cm                                    | 1.40cm                                          |
| 382     | 851              | 2.57cm         | 2.37cm                                    | 3.01cm                                          |
| 2833    | 796              | 1.68cm         | 1.61cm                                    | 2.42cm                                          |
| 2078    | 786              | 1.23cm         | 0.98cm                                    | 1.11cm                                          |

The CSV files generated for each user which contains their ID , Penultimate Layer Output and the Ground Truth can be [accessed here](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/tree/main/Users_MIT/CSVs_MIT_Dinesh)

The code for generating the CSVs file is [link](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/SVR_Sweep/CSV_Creation(Penultimate%20and%20the%20GT).ipynb)

The code for generating unique points is [link](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/SVR_Sweep/Unique_points_creation.py)

#### Analyzing:

Out of all the users the loss is minimum in User ID 2078 which is 1.03cm if we consider all the frames for training and User ID 2301 which is 1.16cm if we consider only 30 unique points for training.The most decrease in loss is for ID 1877 which is around 35%
The average loss before (considering all the frames) is 1.78cm and the loss after SVR was applied was 1.43(Considering all the frames) and 1.76cm respectively(Considering 30 unique frames).This shows that there is an overall improvement of around 20%.

The below results are obtained using  last year’s model on the MIT split as mentioned in the previous section:

| User ID | Number of Frames | MED(MIT Split) | After SVR(3 fold)(Considering all frames) | After SVR(3 fold)(Considering 30 unique points) |
|---------|------------------|----------------|-------------------------------------------|-------------------------------------------------|
| 3183    | 874              | 1.67cm         | 1.41cm                                    | 1.43cm                                          |
| 1877    | 860              | 2.08cm         | 1.35cm                                    | 1.40cm                                          |
| 1326    | 784              | 1.69cm         | 1.23cm                                    | 1.93cm                                          |
| 3140    | 783              | 1.72cm         | 1.38cm                                    | 1.83cm                                          |
| 2091    | 788              | 1.72cm         | 1.54cm                                    | 2.17cm                                          |
| 2301    | 864              | 1.72cm         | 1.2cm                                     | 1.16cm                                          |
| 2240    | 801              | 1.63cm         | 1.22cm                                    | 1.30cm                                          |
| 382     | 851              | 2.67cm         | 2.33cm                                    | 3.13cm                                          |
| 2833    | 796              | 1.71cm         | 1.56cm                                    | 1.88cm                                          |
| 2078    | 786              | 1.22cm         | 1.03cm                                    | 1.38cm                                          |

These results ara obtained after extracting the output of the penultimate layer of the model(1,4) then applying SVR to find the ground truth.

The CSV files generated for each user which contains their ID , Penultimate Layer Output and the Ground Truth can be [accessed here](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/tree/main/Users_MIT)

The code for generating the CSVs file is [link](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/SVR_Sweep/CSV_Creation(Penultimate%20and%20the%20GT).ipynb)

The code for generating unique points is [link](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/SVR_Sweep/Unique_points_creation.py)

#### Analyzing:

Out of all the users the loss is minimum in User ID 2078 which is 1.03cm if we consider all the frames for training and User ID 2301 which is 1.16cm if we consider only 30 unique points for training.The most decrease in loss is for ID 1877 which is around 35%

The average loss before (considering all the frames) is 1.78cm and the loss after SVR was applied was 1.43(Considering all the frames) and 1.76cm respectively(Considering 30 unique frames).This shows that there is an overall improvement of around 20%.

### Comparison of Model(Google Split)

Next the model was trained on google split with the same changes in the parameter.10 individuals have been selected with maximum number of frames in the train test and val set for experimentation.

The results after comparing with previous year’s model is mentioned below.

| User ID | Number of Frames | Current Model(Google Split) | Previous Implemented Model(Google Split) |
|---------|------------------|-----------------------------|------------------------------------------|
| 503     | 965              | 1.32cm                      | 1.50cm                                   |
| 1866    | 1018             | 0.99cm                      | 1.16cm                                   |
| 2459    | 1006             | 0.97cm                      | 1.15cm                                   |
| 1816    | 989              | 0.86cm                      | 1.25cm                                   |
| 3004    | 983              | 1.43cm                      | 1.4cm                                    |
| 3253    | 978              | 0.94cm                      | 1.25cm                                   |
| 1231    | 968              | 1.38cm                      | 1.40cm                                   |
| 2152    | 957              | 1.24cm                      | 1.26cm                                   |
| 2015    | 947              | 1.15cm                      | 1.38cm                                   |
| 1046    | 946              | 1.25cm                      | 1.24cm                                   |



## Experimentations 

The google split was trained with different train test val split using the parameters changes mentioned and the results are as follows:

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
