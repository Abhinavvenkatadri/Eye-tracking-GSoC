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

> MIT Split

> Google Split



#### MIT Split

The MIT Split maintains the train test validation split at a per participant level, same as what GazeCapture does. What this means is that a data from one participant does not appear in more than one of the train/test/val sets. The fact that the same person is not there in all the splits , helps the model to train and generalize well.

Overall after the following conditions have been met, the details regarding frames are as follows:

| **Total Frames** | **Number of Participants** | **Train/Validation/Test** |
|------------------|----------------------------|---------------------------|
| 366,940          | 1,241                      | Train                     |
| 50,946           | 1,219                      | Validation                |
| 83,849           | 1,233                      | Test                      |

Two models have been included for this split.

1.[Current Implemented Model(MIT Split)](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/Checkpoints/CurrentModel_MITSplit.ckpt)

2.[Previous Implemented Model(MIT Split)](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/Checkpoints/Previous_Implemented_Model_MITSplit.ckpt)

The changes in the model will be explained in the further sections


#### Google Split 

Google split their dataset according to the unique ground truth points. What this means is that frames from each participant are present in the train test and validation sets. To ensure no data leaks though, frames related to a particular ground truth point do not appear in more than one set. The split is also a random 70/10/15 train/val/test split compared to a 13 point calibration split.

Two models have been included for this split.

1.[Current Implemented Model(Google Split)](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/Checkpoints/CurrentModel_GoogleSplit.ckpt)

2.[Previous Implemented Model(Google Split)](https://github.com/Abhinavvenkatadri/Eye-tracking-GSoC/blob/main/Checkpoints/Previous_Implemented_Model_GoogleSplit.ckpt)

## Header 2

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
