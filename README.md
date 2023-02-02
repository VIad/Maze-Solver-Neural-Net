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
The inputs are 7x7 images, where each value corresponds to a class, which is either

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
	    maze = = generate_random_maze()
	    X_train[i] = maze
	    Y_train[i] = a_star_solve(maze)


Where each of the Y_train values is a 7x7 matrix which only contains the gates **(3)** and a generated solution path **without the obstacles**. The reason for that is that when training on an Y_train set which also featured the obstacles, more noise was created in the output and larger models were required to reach the same results. Additionally the nature of the task is to only produce the path, so in the Y_train, we don't care about the obstacles.

The dataset used for training / testing is available in `/dataset`
where

 - X.dat, Y.dat - Train set *(~ 63,000 mazes)*
 - X.dat_smol, Y.dat_smol - Cross validation set *( ~ 3000 mazes)*
 - X.dat_test, Y_dat_test - self evident *(~ 32000 mazes)*

## Models, Training & Lessons Learned

![enter image description here](https://i.imgur.com/Jb3ZSNo.png)