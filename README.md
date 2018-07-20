# Indoor-Positioning

## MSc Project: Indoor localization using neural networks

**brief introducton:** 
1. try to use clollected wifi fingerprint to train a rough neural network to roughly predict the location of a mobile device
2. using the accelerometer and the magnetometer data to enhence the time consistency of the location prediction(trajectory continuity)

**current progress:**
- collected data insite, and get the ouput files from the prebuilt android mobile app(more information available:  https://github.com/vradu10/LSR_DataCollection.git). 
- preprocessed the data file, and converted them into standard inputs and outputs that the neural nets required.
- constructed 2 simple neural nets(classification, regression) to predict the location from wifi fingerprint

**to be continue:**
- implement autoencoder layerwise to pretrain the neural nets(make use of the large amount of unmarked wifi data collected previously)
- implement the Hidden Markov Model to enforce time consistency(2 adjacent timestep's location do not differ too much -> tragectory continuity), also think about a way to integrate the accelerometer and magnetometer data to the inputs.


