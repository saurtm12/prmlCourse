# Brief
This is the second assignment of the course: Pattern recognition and Machine Learning
The model needs to detect if an audio file contains bird sound or not.
Kaggle problem description:
"Wildlife monitoring is an important environmental task, where different monitoring systems are deployed in remote areas, in order to collect data of wildlife in vivo. Audio seems to be a very attractive modality, since an audio recording is not affected by different lighting conditions and the, typical, dense vegetation. After having recorded the audio data, then scientists can start process them.

One of the difficulties of processing audio data is that a person has to listen the whole audio recording. This is in contrast with an image, where a person can look an image at one glance, and then annotate it. For example, by just looking an image one can, usually, say quite fast if there is a bird in it. Though, one has to listen an audio file until they listen a bird. The above clearly indicates that if there is a system to quickly decide if an audio segment has or not at least one bird in it, then a first and quick filtering could be performed overt lots of audio samples. Afterwards, the audio samples with bird sounds can be furthered processed by the corresponding scientists.

This competition is about deciding if an audio sample of 10 seconds has a bird sound in it or not, and is based on the Bird Audio Detection task, of the DCASE 2018 Challenge. The evaluation will be based on the same metric as in the Bird Audio Detection task, though a subset of the dataset will be used here."
Kaggle link: https://www.kaggle.com/c/bird-audio-detection

# Team info:
Team name: Duc Hong & Anh Vu
Team member: 
- Nghia Duc Hong H292119
- Duy Anh Vu H294381
- Rank on leaderboard: #17
# Data description 
Training datasets consist of 10 second long audio samples, and associated labels for each audio sample. The labels are in the form "0" or "1", where "1" means that there is a bird sound in the 10 second long audio sample.
For this competition, you have to create a method that takes as an input a 10 second long audio sample, and outputs the probability of having a bird audio in the input audio sample.

You are strongly advised to not use raw audio data as the input to your method. Instead, you should extract some audio features for the audio sample, and use these features as input to your method. A typical feature is the Mel band energies, either as mel band energies or log-scaled (referred to as log-mel band energies). To extract these features, you can use the librosa Python library.

Specifically, you will need the function to load audio and the function to extract the mel band energies. Then, you can just call the log function of numpy, in order to have log-mel band energies.