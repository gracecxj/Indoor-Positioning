# Indoor-Positioning

## MSc Project: Indoor localization using neural networks

**Brief introducton:** 
1. try to use clollected wifi fingerprint to train a rough neural network to roughly predict the location of a mobile device
2. using the accelerometer and the magnetometer data to enhence the time consistency of the location prediction(trajectory continuity)

**Current progress:**
- collected data insite, and get the ouput files from the prebuilt android mobile app(more information available:  https://github.com/vradu10/LSR_DataCollection.git). 
- preprocessed the data file, and converted them into standard inputs and outputs that the neural nets required.
- constructed 2 simple neural nets(classification, regression) to predict the location from wifi fingerprint
- implemented autoencoder layerwise to pretrain the neural nets(make use of the large amount of unlabeled wifi data collected previously)

**Current results visualization:**

![error line classification(64,32,16)](https://github.com/gracecxj/Indoor-Positioning/blob/master/results(gridsize2%2Bauto)/errors_visualization_1.png)
![error line classification(200,200,200)](https://github.com/gracecxj/Indoor-Positioning/blob/master/results(gridsize2%2Bauto)/errors_visualization_1_1.png)

![error line regression(64,32,16)](https://github.com/gracecxj/Indoor-Positioning/blob/master/results(gridsize2%2Bauto)/errors_visualization_2.png)
![error line regression(200，200，200)](https://github.com/gracecxj/Indoor-Positioning/blob/master/results(gridsize2%2Bauto)/errors_visualization_2_1.png)

The foloowing plot is the error in meters results comparision of the above 4 models
![cdf plot of 4 models](https://github.com/gracecxj/Indoor-Positioning/blob/master/results(gridsize2%2Bauto)/CDF(autoencoder).png)

**To be continue:**
- implement the Hidden Markov Model to enforce time consistency(2 adjacent timestep's location do not differ too much -> tragectory continuity), also think about a way to integrate the accelerometer and magnetometer data to the inputs.


