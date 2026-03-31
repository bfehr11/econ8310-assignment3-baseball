# Assignment 3 by Brady Fehr
## Econ 8310 - Business Forecasting

For homework assignment 3, I have built a CNN classifier using a ResNet18 framework that takes a folder of .mov files and returns predictions for each frame of the videos in the folder that indicate whether a baseball is present in the frame. In order to test it's accuracy, you can go to the eval.py file and change folder_path to a path to a folder that contains said .mov files and a corresponding CVAT format annotation file for each .mov file. The accuracy of the model on the frames is displayed as output. In testing, I was able to achieve approximately 99% accuracy.
