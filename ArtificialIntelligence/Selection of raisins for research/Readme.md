# Classification of objects with the optimization method

## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Setup](#setup)
* [Information about author](#information-about-author)

## General info
In this project, I download data from a csv file then, after normalizing and determining the type of data, it saves it in a json file,
then uses a neural network to classify two types of raisins. After classification, some raisins from one variety are selected.
Raisins, the total area of which does not exceed B = 800,000 pixels, can be put into the container. We have an N element set {x1, x2, x3, ..., xN}
(the number of elements of this set will depend on the number of raisins classified into the appropriate class),
in which each element has a specific value cj (circumference size) and size wj (surface area) .

## Requirements:
Python 3.9.0,
Libraries: random, math, json, csv, numpy, sklearn

## Setup
To run this project, run the files in the correct order: SaveFile.py, NeuralNetwork.py, SelectingForReserch

## Information about author
author: Paweł Gradek,
e-mail adress: pawelgradek@gmail.com