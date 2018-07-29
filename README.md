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
- compare different network strcuctures(\[32,64,16\] and \[200,200,200\]). Meantime, see how dropout layer and autoencoder pretrained weights helps the prediction process.

**Current results visualization:**

The following plots is the "error in meters cdf" of different models. More detailed plots(such as error line plot, training curve plot .etc) can be found in results(*) directory. Note: C indicates classification models, while R indicates regression models.

simple vs dropout:

![simple vs dropout](https://github.com/gracecxj/Indoor-Positioning/blob/master/CDF1.png)

simple vs autoencoder:

![simple vs autoencoder](https://github.com/gracecxj/Indoor-Positioning/blob/master/CDF2.png)

autoencoder vs autoencoder+dropout:

![autoencoder vs autoencoder+dropout](https://github.com/gracecxj/Indoor-Positioning/blob/master/CDF3.png)


**To be continue:**
- implement the Hidden Markov Model to enforce time consistency(2 adjacent timestep's location do not differ too much -> tragectory continuity), also think about a way to integrate the accelerometer and magnetometer data to the inputs.


