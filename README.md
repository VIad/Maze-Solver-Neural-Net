# Solving Mazes Using a Neural Network

This is a self assigned 'demo' project in order to understand NNs better.

## The Task

### Given an input maze (7x7)

|M1  |  M2|
|--|--|
| ![enter image description here](https://i.imgur.com/E7H4DHF.png) |  ![enter image description here](https://i.imgur.com/kKIGfuh.png)|


### Produce a path that connects the brightest colored squares (gates)

The actual solution to this problem should be a pathfinding algorithm such as **A*** or ***Dijkstra***. However, that's boring.

So let's start defining things
### The Plan
Build a fully connected [(FC) Deep Network](https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/ch04.html), which recieves the entire maze as an input, learns the features required to solve the maze and output the solution path.
We're going to be using a FC Network, even though a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) might be better for this assignment. In this case, this is an arbitrary restriction and primarily taken because i understand CNNs even less.

### Designing the network
The inputs are 7x7 images (read as arrays), where each value corresponds to a class, which is either

 - Empty passable space
 - Gate (start or end)
 - Obstacle

Given this, we can design our inputs by creating a *NxN* matrix, where each element will contain a number, representing one of those classes, specifically:
 - (**0**) will be empty passable space
 - (**1**) will be an obstacle
 - (**3**) will be a gate

Applying those rules to the **M1** example displayed above yields the following matrix:

    [[0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 1.]
     [3. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 1. 0. 3.]
     [1. 0. 1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0.]]   

*Note that 3 is used to increase value difference between obstacle and gate, which makes a difference during training. More on that later*

Now we just have to flatten the matrix into a single dimension and we get an input vector with a size of 49. So our NN has an input layer with 49 neurons.

As for the **output**, consider both **M1** and **M2**
They each have a maze solution which looks like this

|M1 Solution  | M2 Solution |
|--|--|
| ![enter image description here](https://i.imgur.com/SI4M8zz.png) | ![enter image description here](https://i.imgur.com/NqqmPz7.png) |


Though similar, **M1**'s solution takes up 10 spaces (*it's path could be described by using 10 position vectors of 2 elements each*), whereas **M2**'s solution takes up 8. Our FC net's output layer must have a fixed amount of neurons, but one that can accomodate all possible 7x7 mazes and their respective solutions. 
Therefore, we pick*7x7=49* output layer neurons, the **same size as the input layer**.

Now we have both the input and output layers defined, the rest is figuring out how many hidden layers to use and with what neuron counts. More on that later


## The Dataset
Due to the nature of the problem, the dataset can be created & perfectly engineered to fit the problem's parameters.

The generation part is included in the `~/maze_logic` folder within the project
In a very brief, TL;DR format, this is how the generation works:

    X_train = array_of_size(n_mazes, 7, 7)
    Y_train = array_of_size(n_mazes, 7, 7)
    for i in range 0 to n_mazes:
	    maze = generate_random_maze()
	    X_train[i] = maze
	    Y_train[i] = a_star_solve(maze)


Where each of the Y_train values is a 7x7 matrix which only contains the gates **(3)** and a generated solution path **without the obstacles**. The reason for that is that when training on an Y_train set which also featured the obstacles, more noise was created in the output and larger models were required to reach the same results. Additionally the nature of the task is to only produce the path, so in the Y_train, we don't care about the obstacles.

The dataset used for training / testing is available in `/dataset`
where

 - X.dat, Y.dat - Train set *(~ 63,000 mazes)*
 - X.dat_smol, Y.dat_smol - Cross validation set *( ~ 3000 mazes)*
 - X.dat_test, Y_dat_test - self evident *(~ 32000 mazes)*

## Models & Training

![enter image description here](https://i.imgur.com/GoPThto.png)

### Training
Given that this is a [Regression Model](https://www.imsl.com/blog/what-is-regression-model), all variants of it have been trained to minimize the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) of the set. The `batch_size=64` is set to higher than default to maintain a good Training time / Generalization ratio.

To train the model, i have found that using the [AMSGRAD](https://arxiv.org/pdf/1904.09237.pdf) optimizer seems to bring the model closer to convergence than both SGD and ADAM.

The following plots the loss against the epochs for a pretrained 500 epoch model (*'More layers V3' from the table below*) :

![enter image description here](https://i.imgur.com/kAYbHjT.png)


### What was tried ?

Here's a table sumarizing the models which were attempted.

| # of Parameters | Architecture | Accuracy |
|-----------------|--------------|----------|
| 9,209           | Small / Relu | 38.42%   |
| 11,659          | +2L / Relu   | 40.67%   |
| 11,843          | Sigmoid Pre out / Relu | 56.11% |
| 11,843          | Trained Thick & Sparse Mazes / Relu | 54.30% |
| 27,089          | Sigmoid Out + Dropout regularization| 66.95%   |
| 30,559          | More layers / Linear | 85.47% |
| 61,644          | More layers V2 / Linear| 88.26%  |
| 59,094          | More layers V3 / Linear | 90.31%  |
| 1,519,074       | Large NN (30ep) / Linear| 97.55%  |
| 1,519,074       | Large NN V2 + L2 (300ep) / Linear| 97.93% |


![enter image description here](https://i.imgur.com/iIb411V.png)

Getting to about 86% accuracy was a relatively simple task, a variety of things was tested:

 - Varying dropout
 - Different data generation (train data broken down into ~equal thick and sparse mazes)
 - Trying different optimizers / different output layer functions.

The real uphill battle was getting to the high 90%. As listed above, the models which constistently get into the high 90% have a ridiculous amount of parameters and take stupid long time to train. Colaboratory was heavily utilized for the last 2.

The Large NN models were additionally iteratively tested with different L2 regularization values, dropout rates at varying layers and different amount of epochs (as per the graph Large NN **V1** is trained for 30 epochs, whereas the regularized **V2** is trained for 300). Some of the code for iteratively testing the different models is available in `model/collaboratory/batch_attempt_multiple_models.py`
Ultimately it was a lot of computation for only reliably obtaining less than half a percentage point over the 30 epoch trained Large NN **V1**


### Accuracy
In our regression task accuracy is evaluated by getting a prediction of a maze solution and then tracing it through the actual maze to see if we reach the ending without guessing *(limited to 17 moves)* and without hitting an obstacle. 

## The Models, Visually Compared
![enter image description here](https://i.imgur.com/vg23aRL.png)
![enter image description here](https://i.imgur.com/QIViayX.png)
![enter image description here](https://i.imgur.com/joKYhKj.png)
