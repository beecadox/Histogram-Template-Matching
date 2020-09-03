# Histogram-Template-Matching

The purpose of this query is to implement a function that will object detect an image based on the histogram of a template and as well as the histograms of smaller parts of the image. The implemented algorithm takes each time a piece of the same size image as the selected template. For each individual piece the histogram will be compared with the histogram of the template and if the histograms are too close a frame will be drawn on the image.
The function works for both rgb and grayscale images and the template image may have an odd or even number of rows and / or columns.
