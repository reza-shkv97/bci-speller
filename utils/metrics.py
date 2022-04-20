import numpy as np


def accuracy_score(predictions, target, event):
    score_pos = predictions[:, 1]

    n_character = int(len(score_pos) / 180)

    mean_score = np.zeros((12 * n_character, 1))

    for i in range(n_character):
        for j in range(12):
            c = np.where(event[i, :] == j + 1)
            c = np.array(list(c))
            mean_score[i * 12 + j, 0] = np.mean(score_pos[180 * i + c])

    n_row_col = int(len(mean_score) / 6)
    ind = np.zeros((n_row_col, 1))
    for k in range(n_row_col):
        ind[k, 0] = np.argmax(mean_score[(k * 6):(k + 1) * 6])
    screen_char = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_')

    col = ind[range(0, n_row_col, 2)]
    row = ind[range(1, n_row_col, 2)]
    row_col = col + row * 6

    target_predict = []

    for i in range(len(row_col)):
        target_predict.append(screen_char[int(row_col[i])])

    c = 0
    for i in range(len(target_predict)):

        if target[i] == target_predict[i]:
            c = c + 1
    return c / len(target_predict)
