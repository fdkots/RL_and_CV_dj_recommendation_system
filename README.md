Welcome to my project. 
This is a merge between the RL and CV modules on my curriculum .

I will be starting by the CV module. We will be training a model to capture facial expressions in a DJ set and use that as a metric afterwards for the RL part. 

Let's start with the CV part. 
We will be using open libraries with expressions .

 https://www.format.com/magazine/resources/design/berghain-trainer-website-design . 
 A very interesting site, I feel like the emotion capture was spot on( try it urself) .

 I will be trying to replicate it. I am using this for training : https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

first iteration was very mid, in the lassification of 7 emotions we got a 65% accuracy. 
Due to the nature of the project, it doesnt really make sense to classify in 7 groups. 
I will classify in the following ones : disgust-fear-sad , (neutral) and angry-suprise-happy . 
Why? In a DJ set , what could be deemed as angry could be hype. 

I did some research, and the most prominent architecture for facial recognition is an architecture called EmoNeXt-XLarge 
D. EmoNeXt Final Architecture
The EmoNeXt architecture begins with the inclusion of STN
at the beginning of the network. STN enable the model to
handle variations in scale, rotation, and translation by learning
and applying spatial transformations to facial images.
After passing through the STN, the inputs are then passed
through ConvNeXt’s patchify module. This module down-
samples the input image using a non-overlapping convolution
with a kernel size of 4. The downsampling helps to reduce
the dimensionality of the input and capture relevant features
efficiently.
The downscaled inputs then go through the ConvNeXt
stages. Each stage is followed by a SE block to recalibrate
the feature map before going into the next stage. This recali-
bration enhances the model’s ability to extract discriminative
facial features for accurate emotion recognition. The overall
architecture is illustrated in Figure 4.
By leveraging these techniques, our EmoNeXt model
achieves robust and accurate facial emotion detection, effec-
tively handling variatio

https://arxiv.org/pdf/2501.08199

Input (48×48×1)
     ↓
  [STN] ← handles rotation, scale, translation
     ↓
  [Patchify] Conv 4×4, stride 4 → 12×12×64
     ↓
  [Stage 1] 3× ConvNeXt blocks → SE block  (12×12×64)
     ↓
  [Stage 2] 3× ConvNeXt blocks → SE block  (6×6×128)
     ↓
  [Stage 3] 3× ConvNeXt blocks → SE block  (3×3×256)
     ↓
  [Head] GAP → Dense 256 → Dropout → Dense 3 → Softmax




"Recent studies have highlighted the effectiveness of modern training techniques in significantly improving the performance of deep learning models. In our model training, we employ various advanced strategies to improve the results. We utilize the AdamW optimizer [30] with a learning rate of 1e-4, combined with a cosine decay schedule to enhance convergence. Additionally, we incorporate data augmentation techniques such as RandomCropping and RandomRotation to augment the training data, boosting the model’s ability to generalize. To prevent overfitting, we implement regularization schemes like Stochastic Depth [31] and Label Smoothing [32], which contribute to more robust and generalized models.

To further enhance the performance and address memory constraints, we employ the Exponential Moving Average (EMA) technique [33]. EMA has proven effective in alleviating overfitting, particularly in larger models. Moreover, we adopt Mixed Precision Training [34], a method that reduces memory consumption by almost 2x while accelerating the training process.

In addition, we enhance our model’s capabilities by incorporating weights from pretrained ConvNeXt on the ImageNet-22k dataset [35]. This dataset, known for its vast collection of diverse images, allows our model to leverage a wealth of learned knowledge. To ensure compatibility with the pretrained weights, we resize the images in our training pipeline to 
224
2
, adhering to the established industry practice. This resizing technique enables us to effectively utilize the pretrained weights, resulting in improved performance and enhanced proficiency.

Finally, we trained both the EmoNeXt and ConvNeXt models, encompassing all five sizes (T, S, B, L, and XL), utilizing an Nvidia T4 GPU with 16GB of VRAM. The implementation was done using PyTorch version 2.0.0, and the code is available at: https://github.com/yelboudouri/EmoNeXt

"



PHASE 2 : FRAME SPLIT 

https://www.youtube.com/watch?v=c0-hvjV2A5Y&list=RDc0-hvjV2A5Y&start_radio=1&t=1769s

lets look at this video, its Fred Again performing in London Boiler Room.
Videos like this as you will notice have their camera angle changing. This will be problematic and will mess up our exercise.
So what we need to do is simple, extract a very specific angle of the videos.
I can distinguish 3 primary types of camera angle : 1) camera pointing at DJ and the backstage crowd 2) panned down angle, 3) random crowd camera angle
We will focus on the first. 
How? well we have a landmark we can take as a basis in our analysis, the DJ equipment! 
its always on the bottom half of the screen for the frames we are actually interested in.



1. Problem definition
First define the project in one sentence:
Given the current crowd state and the current song state, the system should choose the next track or next target vibe so as to maximize crowd engagement while keeping transitions smooth.


Perception → CV
Music representation → audio analysis
Decision-making → RL


                    ┌────────────────────────────┐
                    │      TRACK LIBRARY         │
                    │  mp3 / wav / local songs   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   AUDIO FEATURE EXTRACTION │
                    │ librosa / Essentia         │
                    │ - BPM                      │
                    │ - energy                   │
                    │ - spectral features        │
                    │ - optional key             │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │    TRACK FEATURE TABLE     │
                    │ track_id, bpm, energy, ... │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   TRANSITION MODELING      │
                    │ pairwise transition costs  │
                    │ ΔBPM, Δenergy, smoothness  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │     RL ENVIRONMENT         │
                    │ state + action + reward    │
                    └─────────────┬──────────────┘
                                  ▲
                                  │
                                  │
      ┌───────────────────────────┴───────────────────────────┐
      │                                                       │
      │                                                       │
      ▼                                                       │
┌───────────────────────┐                                     │
│    CROWD VIDEO DATA   │                                     │
│ youtube / festival /  │                                     │
│ recorded sessions     │                                     │
└───────────┬───────────┘                                     │
            │                                                 │
            ▼                                                 │
┌───────────────────────┐                                     │
│   VIDEO PREPROCESSING │                                     │
│ ROI crop, stabilization,│                                   │
│ frame sampling         │                                    │
└───────────┬───────────┘                                     │
            │                                                 │
            ▼                                                 │
┌───────────────────────┐                                     │
│    CV FEATURE EXTRACTION│                                   │
│ - motion energy        │                                    │
│ - optional density     │                                    │
│ - optional pose cues   │                                    │
└───────────┬───────────┘                                     │
            │                                                 │
            ▼                                                 │
┌───────────────────────┐                                     │
│  CROWD ENGAGEMENT     │                                     │
│  SIGNAL OVER TIME     │                                     │
│  E(t), D(t)           │                                     │
└───────────┬───────────┘                                     │
            │                                                 │
            ▼                                                 │
┌───────────────────────┐                                     │
│ TRACK-VIDEO ALIGNMENT │─────────────────────────────────────┘
│ match playlist times  │
│ with crowd response   │
└───────────┬───────────┘
            │
            ▼
┌────────────────────────────┐
│   TRACK RESPONSE DATASET   │
│ track + features + crowd   │
│ reaction / reward signal   │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│      TRAINED RL AGENT      │
│ learns transition policy   │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│       FINAL OUTPUT         │
│ recommended next track     │
│ or next BPM / vibe bucket  │
└────────────────────────────┘