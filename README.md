# water-scarcity-hackathon

This project uses a python-developed, u-net convolutional neural network to learn and predict future heatmaps of global water scarcity.

After training, the model can make future projections that look like the following: 

![image test](/docs/demo_heatmap.jpg)

The model trains on two decades of 280x720 resolution global heatmaps and learns using time-series data (predicting the 4th iteration
using the past three iterations of data). There are a total of 18 training samples, and the training inputs are 4D arrays where the
dimensions are as follows:
    1. Sample number
    2. Pixel height
    3. Pixel width
    4. Time series sample number
Each pixel contains a value between 0 - 1 measuring water scarcity where 1 represents a location with a high amount of scarcity. 

The files are divided into its training, inference(testing/projecting), and the heatmap data. There's also a sample demo image to show
a sample year's heatmap.