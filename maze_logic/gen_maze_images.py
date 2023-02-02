import numpy as np
import matplotlib.pyplot as plt
import heapq

# Generate a 7x7 array with random values between 0 and 255
def generate_images(n):
    prepped_array = np.zeros((n, 7, 7))
    image_array = np.zeros((n, 7, 7))
    solution_pts_Y = np.zeros((n, 2, 1))
    for cn in range(n):
        image = np.zeros((7, 7))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if(cn > n / 2): # thick obstacles
                    if np.random.randint(0, 5) >= 3:
                        image[i][j] = 90.0
                else: # Sparse obstacles
                    if np.random.randint(0, 5) == 4:
                        image[i][j] = 90.0
        random_Y_start = np.random.randint(0, 7, 1)
        random_Y_end = np.random.randint(0, 7, 1)
        image[random_Y_start, 0] = 255.0
        image[random_Y_end, 6] = 255.0
        solution_pts_Y[cn][0] = random_Y_start
        solution_pts_Y[cn][1] = random_Y_end
        image_array[cn] = image
        prepped_array[cn] = np.array(convert_to_astar_type(image_array[cn], True))
    return image_array, solution_pts_Y, prepped_array

def generate_solved_mazes(images, points):
    n_mazes = images.shape[0]
    solved = np.zeros((images.shape[0], 7, 7))
    for i in range(n_mazes):
        maze = convert_to_astar_type(images[i], True)
        path_i = np.array(a_star(maze, (int(points[i][0][0]), 0), (int(points[i][1][0]), 6)))
        if path_i is None or len(path_i.shape) == 0:
            continue
        for pp in range(path_i.shape[0] - 1):
            if (pp == 0):
                continue
            images[i][path_i[pp][0]][path_i[pp][1]] = 30.0
        solved[i] = np.array(convert_to_astar_type(images[i], False))
    return images, solved


def convert_to_astar_type(image, include_obstacles):
    maze = np.zeros_like(image).astype(int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j]) == 0.0:
                maze[i][j] = 0
            elif image[i][j] == 90.0 and include_obstacles:
                maze[i][j] = 1
            elif image[i][j] == 30.0:
                maze[i][j] = 2
            elif image[i][j] == 255.0:
                maze[i][j] = 3
    return maze

images, points, prepped = generate_images(20000)


#AUTHOR: ChatGPT
def a_star(maze, start, end):
    # create a priority queue to store the cells to visit
    heap = [(0, start[0], start[1])]
    # create a dictionary to store the came from information
    came_from = {}
    # create a dictionary to store the cost to get to each cell
    cost_so_far = {}
    # initialize the came from and cost so far for the start point
    came_from[start] = None
    cost_so_far[start] = 0
    # while there are still cells to visit
    while heap:
        # get the cell with the lowest cost
        _, x, y = heapq.heappop(heap)
        # if the current cell is the end cell
        if (x, y) == end:
            # create the path by tracing back the came from information
            path = []
            while (x, y) != start:
                path.append((x, y))
                x, y = came_from[(x, y)]
            path.append(start)
            return path[::-1]
        # check the cells on the left, right, top and bottom of the current cell
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if 0 <= x + dx < 7 and 0 <= y + dy < 7 and maze[x+dx][y+dy] != 1:
                # calculate the new cost to get to the cell
                new_cost = cost_so_far[(x, y)] + 1
                # if the cell has not been visited or the new cost is lower than the previous cost
                if (x + dx, y + dy) not in cost_so_far or new_cost < cost_so_far[(x + dx, y + dy)]:
                    # update the came from and cost so far for the cell
                    came_from[(x + dx, y + dy)] = (x, y)
                    cost_so_far[(x + dx, y + dy)] = new_cost
                    # calculate the heuristic cost to get to the end cell
                    heuristic_cost = abs(x + dx - end[0]) + abs(y + dy - end[1])
                    # add the cell to the priority queue
                    heapq.heappush(heap, (new_cost + heuristic_cost, x + dx, y + dy))
    return None


images_solved, solved = generate_solved_mazes(images, points)

print(solved.shape)
print(prepped.shape)

ct_zeros = list()
for i in range(solved.shape[0]):
    if(np.any(solved[i])):
        ct_zeros.append(i)

print(ct_zeros)
solved_act = solved[ct_zeros]
prepped_act = prepped[ct_zeros]

print(len(ct_zeros))
print(prepped_act.shape)
print(solved_act.shape)

print(prepped_act[243])
print(solved_act[243])

fig, axes = plt.subplots(1, 2, figsize=(15,5))
for i in range(1):
    axes[i].imshow(prepped_act[i], cmap='gray')
    axes[i + 1].imshow(solved_act[i], cmap='gray')
plt.show()

np.save("X.dat_smol", prepped_act)
np.save("Y.dat_smol", solved_act)
# maze = convert_to_astar_type(images[0])
# path = np.array(a_star(maze, (int(points[0][0][0]), 0), (int(points[0][1][0]), 6)))
# print(path)
#
# for pp in range(path.shape[0] - 1):
#     if(pp == 0):
#         continue
#     images[0][path[pp][0]][path[pp][1]] = 70.0
#
#
# # increase the size of the figure
# plt.figure(figsize=(3,3))
#
# # Display the image
# plt.imshow(images[0], cmap='gray')
# plt.show()
