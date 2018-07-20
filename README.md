# Indoor-Positioning

## MSc Project

**brief introducton:** 
1. try to use clollected wifi fingerprint to train a rough neural network to roughly predict the location of a mobile device
2. using the accelerometer and the magnetometer data to enhence the time consistency of the location prediction(trajectory continuity)

**current progress:**
- collected data insite, and get the ouput file from the prebuilt android mobile app(more information available:  https://github.com/vradu10/LSR_DataCollection.git). 
- preprocessed the data file, and converted them into standard inputs and outputs that the neural nets required.
- constructed 2 simple neural nets(classification, regression) to predict the location from wifi fingerprint

**to be continue:**
- implement autoencoder layerwise to pretrain the neral nets
- implement the hidden markov model to enforce time consistency(2 adjacent timestep's location do not differ too much -> tragectory continuity)


