import matplotlib.pyplot as plt
import numpy as np
import os
import random

sensor_idx = {
    0: 'wall_4',
    1: 'wall_3_awd',
    2: 'wall_3_wds',
    3: 'wall_3_ads',
    4: 'wall_3_aws',
    5: 'wall_2_aw',
    6: 'wall_2_ad',
    7: 'wall_2_as',
    8: 'wall_2_wd',
    9: 'wall_2_ws',
    10: 'wall_2_ds',
    11: 'wall_1_a',
    12: 'wall_1_w',
    13: 'wall_1_d',
    14: 'wall_1_s',
    15: 'no_wall'
}
sensor_idx_2 = {v: k for k, v in sensor_idx.items()}


movement_matrix = {
    'w': 0.7,
    'a': 0.1,
    'd': 0.1,
    'no': 0.1
}
rotation_matrix = {
    'c': 0.8,
    'plus_90': 0.1,
    'no': 0.1,
}


def create_map(num_maps: int, map_size_x: int, map_size_y: int, num_obstacles: int = 20, obstacle_locations: [int, int] = None) -> np.ndarray:
    # Step 1: Create a 4x10x10 map

    map_data = np.zeros((num_maps, map_size_x, map_size_y))

    # Step 2: Randomly place obstacles (1s) in the first map
    if obstacle_locations is None:
        obstacles = []
        for _ in range(num_obstacles):
            x, y = random.randint(0, map_size_x - 1), random.randint(0, map_size_y - 1)
            obstacles.append((x, y))
            map_data[0, x, y] = 1

        # Step 3: Copy the same obstacles to the other maps
        for i in range(1, num_maps):
            for x, y in obstacles:
                map_data[i, x, y] = 1
    else:
        for i in range(num_maps):
            for x, y in obstacle_locations:
                map_data[i, x, y] = 1
    return map_data


def show_maps(map_data: np.ndarray, num_maps: int, map_size_x: int, map_size_y: int, main_title: str = None, img_name: str = None):
    if main_title is None:
        main_title = f'{map_size_x}x{map_size_y} Map'

    # Step 4: Visualize the maps side by side with grid lines on integer numbers
    robot_direction = ['W', 'N', 'E', 'S']
    titles = [f'{main_title} ({robot_direction[i]})' for i in range(num_maps)]
    fig, axes = plt.subplots(1, num_maps, figsize=(20, 5))
    for i in range(num_maps):
        ax = axes[i]
        ax.imshow(map_data[i], cmap='Greys', interpolation='none')
        ax.set_title(titles[i])
        ax.set_xticks(np.arange(0.5, map_size_x, 1))
        ax.set_yticks(np.arange(0.5, map_size_y, 1))
        ax.grid(which='both', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='both', bottom=False, left=False)
        ax.axis('on')
    if img_name is not None:
        fig.savefig(img_name)
    plt.show()


def show_true_map_with_robot(
        map_data: np.ndarray,
        map_size_x: int,
        map_size_y: int,
        direction: str,
        main_title: str = None,
        img_name: str = None
):
    if main_title is None:
        main_title = 'Map with Robot'

    direction_arrows = {
        'N': '↑',
        'S': '↓',
        'E': '→',
        'W': '←'
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.imshow(map_data, cmap='Greys', interpolation='none')
    ax.set_title(main_title)
    ax.set_xticks(np.arange(0.5, map_size_x, 1))
    ax.set_yticks(np.arange(0.5, map_size_y, 1))
    ax.grid(which='both', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False)
    ax.axis('on')

    # Annotate the grid with the values
    for x in range(map_size_x):
        for y in range(map_size_y):
            value = map_data[x, y]
            if value == 2:
                ax.text(y, x, direction_arrows[direction], ha='center', va='center', color='red', fontsize=12)
            # elif value != 1:
            #     ax.text(y, x, f'{value:.2f}', ha='center', va='center', color='black', fontsize=6)

    fig.colorbar(cax, ax=ax, shrink=0.5)
    if img_name is not None:
        fig.savefig(img_name)
    plt.show()


def show_localization_map(
        map_data: np.ndarray,
        num_maps: int,
        map_size_x: int,
        map_size_y: int,
        main_title: str = None,
        img_name: str = None
):
    if main_title is None:
        main_title = f'{map_size_x}x{map_size_y} Map with Numbers'
    robot_direction = ['W (←)', 'N (↑)', 'E (→)', 'S (↓)']
    titles = [f'{robot_direction[i]}' for i in range(num_maps)]
    fig, axes = plt.subplots(1, num_maps, figsize=(20, 5))
    fig.suptitle(main_title, fontsize=16)
    # Find the min and max values excluding 1
    min_val = np.min(map_data[map_data != 1])
    max_val = np.max(map_data[map_data != 1])
    for i in range(num_maps):
        ax = axes[i]
        cax = ax.imshow(map_data[i], cmap='cool', interpolation='none', vmin=min_val, vmax=max_val+0.01) # spring RdYlGn Spectral
        ax.set_title(titles[i])
        ax.set_xticks(np.arange(0.5, map_size_x, 1))
        ax.set_yticks(np.arange(0.5, map_size_y, 1))
        ax.grid(which='both', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='both', bottom=False, left=False)
        ax.axis('on')

        # Annotate the grid with the values
        for x in range(map_size_x):
            for y in range(map_size_y):
                value = map_data[i, x, y]
                if value != 1:
                    ax.text(y, x, f'{value:.4f}', ha='center', va='center', color='black', fontsize=6)
                else:
                    ax.add_patch(plt.Rectangle((y-0.5, x-0.5), 1, 1, fill=True, color='black'))

    fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.5)
    if img_name is not None:
        fig.savefig(img_name)
    plt.show()


def add_robot(map_data: np.ndarray, x: int, y: int) -> np.ndarray:
    map_data_with_robot = map_data.copy()
    map_data_with_robot[:, x, y] = 2
    return map_data_with_robot


def find_robot(map_data: np.ndarray) -> tuple:
    for x in range(map_data.shape[0]):
        for y in range(map_data.shape[1]):
            if map_data[x, y] == 2:
                return x, y
    return -1, -1


def move(map_data: np.ndarray, map_size_x: int, map_size_y: int, direction: str) -> np.ndarray:
    """
    Move the robot in the given direction
    Args:
        map_data:
        map_size:
        direction: is one of ['A', 'W', 'D', 'S']

    Returns:

    """
    robot_direction = ['W', 'N', 'E', 'S']
    for idx, map_direction in enumerate(map_data):
        robot_location = find_robot(map_direction)
        if robot_direction[idx] == 'W':
            if direction == 'W' and robot_location[1] - 1 >= 0 \
                    and map_data[idx, robot_location[0], robot_location[1] - 1] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0], robot_location[1] - 1] = 2
            elif direction == 'S' and robot_location[1] + 1 < map_size_y \
                    and map_data[idx, robot_location[0], robot_location[1] + 1] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0], robot_location[1] + 1] = 2
            elif direction == 'A' and robot_location[0] + 1 < map_size_x \
                    and map_data[idx, robot_location[0] + 1, robot_location[1]] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0] + 1, robot_location[1]] = 2
            elif direction == 'D' and robot_location[0] - 1 >= 0 and \
                    map_data[idx, robot_location[0] - 1, robot_location[1]] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0] - 1, robot_location[1]] = 2

        elif robot_direction[idx] == 'N':
            if direction == 'W' and robot_location[0] - 1 >= 0 and \
                    map_data[idx, robot_location[0] - 1, robot_location[1]] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0] - 1, robot_location[1]] = 2
            elif direction == 'S' and robot_location[0] + 1 < map_size_x and \
                    map_data[idx, robot_location[0] + 1, robot_location[1]] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0] + 1, robot_location[1]] = 2
            elif direction == 'A' and robot_location[1] - 1 >= 0 and \
                    map_data[idx, robot_location[0], robot_location[1] - 1] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0], robot_location[1] - 1] = 2
            elif direction == 'D' and robot_location[1] + 1 < map_size_y and \
                    map_data[idx, robot_location[0], robot_location[1] + 1] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0], robot_location[1] + 1] = 2

        elif robot_direction[idx] == 'E':
            if direction == 'W' and robot_location[1] + 1 < map_size_y and \
                    map_data[idx, robot_location[0], robot_location[1] + 1] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0], robot_location[1] + 1] = 2
            elif direction == 'S' and robot_location[1] - 1 >= 0 and \
                    map_data[idx, robot_location[0], robot_location[1] - 1] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0], robot_location[1] - 1] = 2
            elif direction == 'A' and robot_location[0] - 1 >= 0 and \
                    map_data[idx, robot_location[0] - 1, robot_location[1]] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0] - 1, robot_location[1]] = 2
            elif direction == 'D' and robot_location[0] + 1 < map_size_x and \
                    map_data[idx, robot_location[0] + 1, robot_location[1]] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0] + 1, robot_location[1]] = 2

        elif robot_direction[idx] == 'S':
            if direction == 'W' and robot_location[0] + 1 < map_size_x and \
                    map_data[idx, robot_location[0] + 1, robot_location[1]] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0] + 1, robot_location[1]] = 2
            elif direction == 'S' and robot_location[0] - 1 >= 0 and \
                    map_data[idx, robot_location[0] - 1, robot_location[1]] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0] - 1, robot_location[1]] = 2
            elif direction == 'A' and robot_location[1] + 1 < map_size_y and \
                    map_data[idx, robot_location[0], robot_location[1] + 1] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0], robot_location[1] + 1] = 2
            elif direction == 'D' and robot_location[1] - 1 >= 0 and \
                    map_data[idx, robot_location[0], robot_location[1] - 1] != 1:
                map_data[idx, robot_location[0], robot_location[1]] = 0
                map_data[idx, robot_location[0], robot_location[1] - 1] = 2

    return map_data


def rotate(map_data: np.ndarray, direction: str) -> np.ndarray:
    """
    Rotate the robot in the given direction
    Args:
        map_data:
        direction: is one of ['L', 'R']

    Returns:

    """
    new_map_data = map_data.copy()
    if direction == 'R':
        new_map_data = np.roll(map_data, 1, axis=0)
    elif direction == 'L':
        new_map_data = np.roll(map_data, -1, axis=0)
    return new_map_data


def create_sensor_matrix(size: int, number_on_main: float = 0.8) -> np.ndarray:
    matrix = np.zeros((size, size))
    for i in range(size):
        row = np.random.rand(size)
        row[i] = 0.0  # Temporarily set the diagonal element to 0
        row_sum = np.sum(row)
        row = row / row_sum * (1 - number_on_main)  # Normalize the row to sum to 0.2
        row[i] = number_on_main  # Set the main diagonal to 0.8
        matrix[i] = row
    matrix = np.round(matrix, 5)
    return matrix


def get_wall_results(
        map_data: np.ndarray,
        robot_direction: str,
        location_x: int,
        location_y: int,
        map_size_x: int,
        map_size_y: int
) -> str:
    walls_N = {
        'w': False,
        'd': False,
        's': False,
        'a': False
    }
    if location_y - 1 < 0 or (location_y - 1 >= 0 and map_data[location_x, location_y - 1] == 1):
        walls_N['a'] = True
    if location_x + 1 >= map_size_x or (location_x + 1 < map_size_x and map_data[location_x + 1, location_y] == 1):
        walls_N['s'] = True
    if location_y + 1 >= map_size_y or (location_y + 1 < map_size_y and map_data[location_x, location_y + 1] == 1):
        walls_N['d'] = True
    if location_x - 1 < 0 or (location_x - 1 >= 0 and map_data[location_x - 1, location_y] == 1):
        walls_N['w'] = True

    if robot_direction == 'W':
        walls = {
            'w': walls_N['a'],
            'd': walls_N['w'],
            's': walls_N['d'],
            'a': walls_N['s']
        }
    elif robot_direction == 'E':
        walls = {
            'w': walls_N['d'],
            'd': walls_N['s'],
            's': walls_N['a'],
            'a': walls_N['w']
        }
    elif robot_direction == 'S':
        walls = {
            'w': walls_N['s'],
            'd': walls_N['a'],
            's': walls_N['w'],
            'a': walls_N['d']
        }
    else:
        walls = {
            'w': walls_N['w'],
            'd': walls_N['d'],
            's': walls_N['s'],
            'a': walls_N['a']
        }
    wall_count = sum(value for value in walls.values())
    if wall_count == 0:
        wall_str = 'no_wall'
    elif wall_count == 4:
        wall_str = 'wall_4'
    else:
        true_keys = [key for key, value in walls.items() if value]
        if wall_count == 1:
            wall_str = f'wall_1_{true_keys[0]}'
        else:
            sequence = 'awds'
            sorted_true_keys = ''.join(sorted(true_keys, key=lambda x: sequence.index(x)))
            wall_str = f'wall_{wall_count}_{sorted_true_keys}'
    if wall_str not in sensor_idx_2.keys():
        print("Error: Wall string not found in the sensor matrix")

    return wall_str

def get_other_movement_locations(
        map_data: np.ndarray,
        robot_direction: str,
        location_x: int,
        location_y: int,
        map_size_x: int, map_size_y: int,
        robot_movement: str
):
    # TODO return type
    movement_N = {
        'w': [-1, -1],
        'd': [-1, -1],
        'a': [-1, -1],
        's': [-1, -1]
    }
    if (robot_movement == 'W' and robot_direction == 'N') or \
            (robot_movement == 'A' and robot_direction == 'E') or \
            (robot_movement == 'D' and robot_direction == 'W') or \
            (robot_movement == 'S' and robot_direction == 'S'):
        chosen_direction = 'N'
    if (robot_movement == 'W' and robot_direction == 'W') or \
            (robot_movement == 'A' and robot_direction == 'N') or \
            (robot_movement == 'D' and robot_direction == 'S') or \
            (robot_movement == 'S' and robot_direction == 'E'):
        chosen_direction = 'W'
    if (robot_movement == 'W' and robot_direction == 'S') or \
            (robot_movement == 'A' and robot_direction == 'W') or \
            (robot_movement == 'D' and robot_direction == 'E') or \
            (robot_movement == 'S' and robot_direction == 'N'):
        chosen_direction = 'S'
    if (robot_movement == 'W' and robot_direction == 'E') or \
            (robot_movement == 'A' and robot_direction == 'S') or \
            (robot_movement == 'D' and robot_direction == 'N') or \
            (robot_movement == 'S' and robot_direction == 'W'):
        chosen_direction = 'E'

    if location_y - 1 >= 0 and (map_data[location_x, location_y - 1] != 1):
        movement_N['a'] = [location_x, location_y - 1]
    if location_y + 1 < map_size_y and (map_data[location_x, location_y + 1] != 1):
        movement_N['d'] = [location_x, location_y + 1]
    if location_x - 1 >= 0 and map_data[location_x - 1, location_y] != 1:
        movement_N['w'] = [location_x - 1, location_y]
    if location_x + 1 < map_size_x and map_data[location_x + 1, location_y] != 1:
        movement_N['s'] = [location_x + 1, location_y]

    if chosen_direction == 'W':
        movement_side = {
            'w': movement_N['a'],
            'd': movement_N['w'],
            's': movement_N['d'],
            'a': movement_N['s']
        }
    elif chosen_direction == 'E':
        movement_side = {
            'w': movement_N['d'],
            'd': movement_N['s'],
            's': movement_N['a'],
            'a': movement_N['w']
        }
    elif chosen_direction == 'S':
        movement_side = {
            'w': movement_N['s'],
            'd': movement_N['a'],
            's': movement_N['w'],
            'a': movement_N['d']
        }
    else:
        movement_side = {
            'w': movement_N['w'],
            'd': movement_N['d'],
            's': movement_N['s'],
            'a': movement_N['a']
        }
    movement_side_final = {
            'w': movement_side['s'],
            'd': movement_side['a'],
            's': movement_side['w'],
            'a': movement_side['d']
        }
    movement_side_final.pop('s')

    return movement_side_final

def update_localization_map(
        map_data_real: np.ndarray,
        map_data_location: np.ndarray,
        robot_direction: str,
        map_size_x: int, map_size_y: int
) -> np.ndarray:
    robot_location_x, robot_location_y = find_robot(map_data_real)
    robot_walls = get_wall_results(
        map_data=map_data_real, robot_direction=robot_direction, location_x=robot_location_x,
        location_y=robot_location_y, map_size_x=map_size_x, map_size_y=map_size_y
    )
    true_wall_idx = sensor_idx_2[robot_walls]
    map_no_robot = map_data_real.copy()
    map_no_robot[robot_location_x, robot_location_y] = 0
    wnes_directions = ['W', 'N', 'E', 'S']
    for direction in range(len(wnes_directions)):
        for x_location in range(map_size_x):
            for y_location in range(map_size_y):
                if map_data_location[direction, x_location, y_location] != 1:
                    map_location_walls = get_wall_results(
                        map_data=map_no_robot, robot_direction=wnes_directions[direction], location_x=x_location,
                        location_y=y_location,
                        map_size_x=map_size_x, map_size_y=map_size_y
                    )
                    map_wall_idx = sensor_idx_2[map_location_walls]
                    sensor_matrix_value = sensor_matrix[true_wall_idx][map_wall_idx]
                    # cur value * sensor matrix value
                    map_data_location[direction, x_location, y_location] *= sensor_matrix_value
    # normalization
    sum_values = np.round(np.sum(map_data_location[map_data_location != 1]), 5)
    map_data_location[map_data_location != 1] /= sum_values
    return sensor_matrix


def init_localization_map(map_data: np.ndarray) -> np.ndarray:
    loc_map = map_data.copy()
    zero_count = np.count_nonzero(map_data == 0)
    init_value = 1 / zero_count
    loc_map[map_data == 0] = init_value
    return loc_map


def get_current_true_map(map_data: np.ndarray, direction: str) -> np.ndarray:
    if direction == 'N':
        return map_data[1]
    elif direction == 'E':
        return map_data[2]
    elif direction == 'W':
        return map_data[0]
    elif direction == 'S':
        return map_data[3]
    else:
        return map_data[0]


def update_true_direction(current_direction: str, rotation: str) -> str:
    compass = ['W', 'N', 'E', 'S']
    current_index = compass.index(current_direction)
    if rotation == 'R':
        new_index = (current_index + 1) % 4
    elif rotation == 'L':
        new_index = (current_index - 1) % 4
    return compass[new_index]

def update_location_after_movement(
        map_data_location: np.ndarray,
        map_size_x: int, map_size_y: int,
        chosen_movement: str # one of AWDS
) -> np.ndarray:
    wnes_directions = ['W', 'N', 'E', 'S']
    map_data_final = map_data_location.copy()
    for direction in range(len(wnes_directions)):
        for x_location in range(map_size_x):
            for y_location in range(map_size_y):
                if map_data_location[direction, x_location, y_location] != 1:
                    formula = map_data_location[direction, x_location, y_location] * movement_matrix['no']
                    movement_side = get_other_movement_locations(
                        map_data=map_data_location[direction], robot_direction=wnes_directions[direction],
                        location_x=x_location, location_y=y_location, map_size_x=map_size_x, map_size_y=map_size_y,
                        robot_movement=chosen_movement
                    )
                    for key, value in movement_side.items():
                        if value != [-1, -1]:
                            formula += map_data_location[direction, value[0], value[1]] * movement_matrix[key]

                    map_data_final[direction, x_location, y_location] = formula
    # normalization
    sum_values = np.round(np.sum(map_data_final[map_data_final != 1]), 5)
    map_data_final[map_data_final != 1] /= sum_values

    return map_data_final


def update_location_after_rotation(
        map_data_location: np.ndarray,
        rotation: str,
        map_size_x: int, map_size_y: int
) -> np.ndarray:
    wnes_directions = ['W', 'N', 'E', 'S']
    map_data_final = map_data_location.copy()
    for direction in range(len(wnes_directions)):
        for x_location in range(map_size_x):
            for y_location in range(map_size_y):
                if map_data_location[direction, x_location, y_location] != 1:

                    formula = map_data_location[direction, x_location, y_location] * rotation_matrix['no']
                    direction_letter = wnes_directions[direction]
                    if direction_letter == 'W':
                        plus_90 = 'E'
                        if rotation == 'L':
                            correct = 'N'
                        else:
                            correct = 'S'
                    elif direction_letter == 'N':
                        plus_90 = 'S'
                        if rotation == 'L':
                            correct = 'E'
                        else:
                            correct = 'W'
                    elif direction_letter == 'E':
                        plus_90 = 'W'
                        if rotation == 'L':
                            correct = 'S'
                        else:
                            correct = 'N'
                    elif direction_letter == 'S':
                        plus_90 = 'N'
                        if rotation == 'L':
                            correct = 'W'
                        else:
                            correct = 'E'
                    plus_90_idx = wnes_directions.index(plus_90)
                    correct_idx = wnes_directions.index(correct)
                    formula += map_data_location[plus_90_idx, x_location, y_location] * rotation_matrix['plus_90']
                    formula += map_data_location[correct_idx, x_location, y_location] * rotation_matrix['c']
                    map_data_final[direction, x_location, y_location] = formula
    # normalization
    sum_values = np.sum(map_data_final[map_data_final != 1])
    map_data_final[map_data_final != 1] /= sum_values

    return map_data_final


def get_actual_movement(chosen_movement: str, chosen_probability: str) -> str:
    return_movement = chosen_movement
    if chosen_probability != 'w' and chosen_probability != 'no':
        if chosen_probability == 'a':
            if chosen_movement == 'W':
                return_movement = 'A'
            elif chosen_movement == 'A':
                return_movement = 'S'
            elif chosen_movement == 'D':
                return_movement = 'W'
            elif chosen_movement == 'S':
                return_movement = 'D'
        elif chosen_probability == 'd':
            if chosen_movement == 'W':
                return_movement = 'D'
            elif chosen_movement == 'A':
                return_movement = 'W'
            elif chosen_movement == 'D':
                return_movement = 'S'
            elif chosen_movement == 'S':
                return_movement = 'A'
    if chosen_probability == 'no':
        return_movement = 'no'

    return return_movement


if __name__ == '__main__':
    img_folder = './images'
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    directions = 4
    size_horizontal = 10
    size_vertical = 10
    sensor_matrix = create_sensor_matrix(size=len(sensor_idx), number_on_main=0.6)
    # obstacle_locations = [(0, 1), (2, 2)]
    new_map = create_map(num_maps=directions, map_size_x=size_horizontal, map_size_y=size_vertical, num_obstacles=40)#, obstacle_locations=obstacle_locations)
    localization_map = init_localization_map(new_map)
    show_localization_map(
        map_data=localization_map, num_maps=directions, map_size_x=size_horizontal, map_size_y=size_vertical,
        main_title='Localization Map', img_name=f'{img_folder}/0_localization.png')
    # show_maps(map_data=new_map, num_maps=directions, map_size=size_x_y)
    user_chosen_robot_position = input("Choose robot position x, y: ")
    rob_pos_x, rob_pos_y = user_chosen_robot_position.split(',')
    rob_pos_x_int = int(rob_pos_x.strip())
    rob_pos_y_int = int(rob_pos_y.strip())
    map_w_robot = add_robot(new_map, rob_pos_x_int, rob_pos_y_int)
    # show_maps(
    #     map_data=map_w_robot, num_maps=directions, map_size_x=size_horizontal, map_size_y=size_vertical,
    #     main_title='map starting with robot'
    # )
    possible_movement = ['A', 'W', 'D', 'S']
    possible_rotations = ['R', 'L']

    true_direction = random.choice(['W', 'N', 'E', 'S'])
    real_map = get_current_true_map(map_data=map_w_robot, direction=true_direction)
    show_true_map_with_robot(
        map_data=real_map, map_size_x=size_horizontal, map_size_y=size_vertical, direction=true_direction,
        main_title='True Map with robot', img_name=f'{img_folder}/0_map_robot.png'
    )

    going_on = True
    updated_map = map_w_robot
    update_localization_map(map_data_real=real_map, map_data_location=localization_map,
                            robot_direction=true_direction, map_size_x=size_horizontal, map_size_y=size_vertical)
    show_localization_map(
        map_data=localization_map, num_maps=directions, map_size_x=size_horizontal, map_size_y=size_vertical,
        main_title=f'Localization Map after start - sensors', img_name=f'{img_folder}/0_sensors_localization.png'
    )
    movement_count = 0
    movement_count += 1
    random_keys_m = []
    for key_m, prob_m in movement_matrix.items():
        random_keys_m.extend([key_m] * int(prob_m * 10))
    random_keys_r = []
    for key_r, prob_r in rotation_matrix.items():
        random_keys_r.extend([key_r] * int(prob_r * 10))
    while going_on:
        movement = input("Movement of the robot ['A', 'W', 'D', 'S'] for moving, ['R', 'L'] "
                         "for rotation, '-1' for stop: ")
        if movement in possible_movement:
            actual_movement_prob = random.choice(random_keys_m)
            real_movement = get_actual_movement(chosen_movement=movement, chosen_probability=actual_movement_prob)
            print(f'Users movement: {movement}, actual_movement_prob: {actual_movement_prob}, '
                  f'real_movement: {real_movement}')
            if actual_movement_prob != 'no':
                updated_map = move(
                    updated_map, map_size_x=size_horizontal, map_size_y=size_vertical, direction=real_movement
                )
                # show_maps(
                #     map_data=updated_map, num_maps=directions, map_size_x=size_horizontal, map_size_y=size_vertical,
                #     main_title=f'move {movement}'
                # )

            real_map = get_current_true_map(map_data=updated_map, direction=true_direction)
            show_true_map_with_robot(map_data=real_map, map_size_x=size_horizontal, map_size_y=size_vertical, direction=true_direction,
                                     main_title=f'True Map with robot (c: {movement}, t: {real_movement})', img_name=f'{img_folder}/{movement_count}_map_robot.png')
            localization_map = update_location_after_movement(map_data_location=localization_map, map_size_x=size_horizontal, map_size_y=size_vertical, chosen_movement=movement)
            show_localization_map(map_data=localization_map, num_maps=directions, map_size_x=size_horizontal, map_size_y=size_vertical,
                                  main_title=f'Localization Map after (c: {movement}, t: {real_movement}) - moving', img_name=f'{img_folder}/{movement_count}_localization.png')
        elif movement in possible_rotations:
            actual_movement_prob = random.choice(random_keys_r)
            print(f'Users movement: {movement}, actual_movement_prob: {actual_movement_prob},')
            if actual_movement_prob != 'no':
                updated_map = rotate(updated_map, direction=movement)
                # show_maps(
                #     map_data=updated_map, num_maps=directions, map_size_x=size_horizontal, map_size_y=size_vertical,
                #     main_title=f'rotation {movement}'
                # )

                # localization_map = rotate(localization_map, direction=movement)
                true_direction = update_true_direction(current_direction=true_direction, rotation=movement)
                real_map = get_current_true_map(map_data=updated_map, direction=true_direction)
                if actual_movement_prob == 'plus_90':
                    updated_map = rotate(updated_map, direction=movement)
                    true_direction = update_true_direction(current_direction=true_direction, rotation=movement)
                    real_map = get_current_true_map(map_data=updated_map, direction=true_direction)
            show_true_map_with_robot(
                map_data=real_map, map_size_x=size_horizontal, map_size_y=size_vertical, direction=true_direction,
                main_title=f'True Map with robot (c: {movement} t: {actual_movement_prob})',
                img_name=f'{img_folder}/{movement_count}_map_robot.png'
            )

            localization_map = update_location_after_rotation(map_data_location=localization_map, rotation=movement,
                                                              map_size_x=size_horizontal, map_size_y=size_vertical)
            show_localization_map(map_data=localization_map, num_maps=directions, map_size_x=size_horizontal,
                                  map_size_y=size_vertical,
                                  main_title=f'Localization Map after {movement} ({actual_movement_prob}) - rotating',
                                  img_name=f'{img_folder}/{movement_count}_localization.png')
        else:
            if movement == '-1':
                going_on = False
            else:
                print(f'Wrong command {movement}')
        update_localization_map(map_data_real=real_map, map_data_location=localization_map,
                                robot_direction=true_direction, map_size_x=size_horizontal, map_size_y=size_vertical)
        show_localization_map(
            map_data=localization_map, num_maps=directions, map_size_x=size_horizontal, map_size_y=size_vertical,
            main_title=f'Localization Map after {movement} - sensors',
            img_name=f'{img_folder}/{movement_count}_sensors_localization.png'
        )



