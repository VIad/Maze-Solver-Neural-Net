import numpy as np


def evaluate_found_path(answer, X_test):
    start = find_gate(answer, 0.6, True)
    end = find_gate(answer, 0.6, False)
    if start is None or end is None:
        return False
    current = (start[0], start[1])
    visited = set()
    visited.add(current)
    # If not found within 17 moves, consider not solved
    for i in range(17):
        brightest_neighbour = find_brightest_neighbour(answer, current, visited, X_test)
        if brightest_neighbour is None:
            return False
        current = (brightest_neighbour[0], brightest_neighbour[1])
        visited.add(current)
        if current == end:
            return True
    return False


def find_gate(answer, epsillon, start):
    for row in range(7):
        if start:
            if answer[row][0] > (3 - epsillon):
                return row, 0
        else:
            if answer[row][6] > (3 - epsillon):
                return row, 6
    return None


def find_brightest_neighbour(answer, position, visited, X_test):
    ind_up = (max(position[0] - 1, 0), position[1])
    ind_down = (min(position[0] + 1, 6), position[1])
    ind_left = (position[0], max(position[1] - 1, 0))
    ind_right = (position[0], min(position[1] + 1, 6))

    if ind_up in visited:
        value_up = -1
    else:
        value_up = answer[ind_up[0]][ind_up[1]]
    if ind_down in visited:
        value_down = -1
    else:
        value_down = answer[ind_down[0]][ind_down[1]]
    if ind_left in visited:
        value_left = -1
    else:
        value_left = answer[ind_left[0]][ind_left[1]]
    if ind_right in visited:
        value_right = -1
    else:
        value_right = answer[ind_right[0]][ind_right[1]]

    # Find the maximum value among the neighbours and return the corresponding index
    max_value = max(value_up, value_down, value_left, value_right)
    if max_value == -1:
        return None
    if max_value == value_up:
        # Brightest is up, but up is a wrong choice, failed test
        if X_test[ind_up[0]][ind_up[1]] == 1.0:
            return None
        return ind_up
    elif max_value == value_down:
        # Brightest is down, but down is a wrong choice, failed test
        if X_test[ind_down[0]][ind_down[1]] == 1.0:
            return None
        return ind_down
    elif max_value == value_left:
        # Brightest is left, but left is a wrong choice, failed test
        if X_test[ind_left[0]][ind_left[1]] == 1.0:
            return None
        return ind_left
    elif max_value == value_right:
        # Brightest is right, but right is a wrong choice, failed test
        if X_test[ind_right[0]][ind_right[1]] == 1.0:
            return None
        return ind_right


def evaluate_total_predictions_set(Y_hat, X_test):
    preds_ = np.zeros(Y_hat.shape[0], dtype=bool)
    for index in range(preds_.size):
        preds_[index] = evaluate_found_path(Y_hat[index].reshape(7, 7), X_test[index].numpy().reshape(7,7))

    total = preds_.shape[0]
    correct = preds_[preds_ == True].shape[0]
    correct_percentage = (correct / total) * 100
    print("\n----------MODEL SUMMARY-----------")
    print("TOTAL SET SIZE: ", total)
    print("CORRECT GUESSES: ", correct)
    print("TOTALING TO ACCURACY%: ", correct_percentage)
    print("------------------------------------\n\n")

    print(preds_.shape[0], preds_[preds_ == True].shape[0])
    return correct_percentage
