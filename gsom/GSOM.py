import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math

data_filename = "example/data/zoo.txt".replace('\\', '/')


class GSOM:

    def __init__(self, spred_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8,
                 max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=30000, random_state=42):
        """
        GSOM structure:
        keep dictionary to x,y coordinates and numpy array to keep weights
        :param spred_factor: spread factor of GSOM graph
        :param dimensions: weight vector dimensions
        :param distance: distance method: support scipy.spatial.distance.cdist
        :param initialize: weight vector initialize method
        :param learning_rate: initial training learning rate of weights
        :param smooth_learning_factor: smooth learning factor to change the initial smooth learning rate from training
        :param max_radius: maximum neighbourhood radius
        :param FD: spread weight value
        :param r: learning rate update value
        :param alpha: learning rate update value
        :param initial_node_size: initial node allocation in memory
        """
        self.initial_node_size = initial_node_size
        self.node_count = 0
        self.map = {}
        self.node_list = np.zeros((self.initial_node_size, dimensions))
        self.node_coordinate = np.zeros((self.initial_node_size, 2), dtype=np.int32)
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.float64)
        self.spred_factor = spred_factor
        self.groth_threshold = -dimensions * math.log(self.spred_factor)
        self.FD = FD
        self.R = r
        self.ALPHA = alpha
        self.LAMBDA = np.clip(1 - self.spred_factor, 0.2, 0.9)
        self.dimentions = dimensions
        self.distance = distance
        self.initialize = initialize
        self.learning_rate = learning_rate
        self.smooth_learning_factor = smooth_learning_factor
        self.max_radius = max_radius
        self.random_state = random_state
        self.initialize_GSOM()
        self.node_labels = None
        self.output = None
        self.predictive = None
        self.active = None
        self.sequence_weights = None
        
        # Pre-compute coordinate index for faster lookups
        self._coord_to_index = {}
        self._update_coord_index()

    def _update_coord_index(self):
        """Update coordinate to index mapping"""
        self._coord_to_index = self.map.copy()

    def initialize_GSOM(self):
        np.random.seed(self.random_state)
        self.insert_node_with_weights(1, 1)
        self.insert_node_with_weights(1, 0)
        self.insert_node_with_weights(0, 1)
        self.insert_node_with_weights(0, 0)

    def insert_new_node(self, x, y, weights):
        if self.node_count >= self.initial_node_size:
            # Resize arrays dynamically
            new_size = self.initial_node_size * 2
            self.node_list = np.resize(self.node_list, (new_size, self.dimentions))
            self.node_coordinate = np.resize(self.node_coordinate, (new_size, 2))
            self.node_errors = np.resize(self.node_errors, new_size)
            self.initial_node_size = new_size
            
        self.map[(x, y)] = self.node_count
        self.node_list[self.node_count] = weights
        self.node_coordinate[self.node_count] = [x, y]
        self.node_count += 1

    def insert_node_with_weights(self, x, y):
        if self.initialize == 'random':
            node_weights = np.random.rand(self.dimentions)
        else:
            print("initialize method not support")
        self.insert_new_node(x, y, node_weights)

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        #alternative decay function for radius
        # radius(e+1) = 1+ (radius(0)-1)*(1 - e/E) whre E is total iterations, e is current iteration
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(- iteration / time_constant)
    
    def _get_lattice_neighbors(self, x, y, radius):
        """Optimized neighbor search"""
        neighbors = []
        # We use manhattan distance for diamond neighbourhood
        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if (i, j) in self.map and (i, j) != (x, y):
                    if abs(i - x) + abs(j - y) <= radius:
                        neighbors.append((i, j))
        return neighbors

    def _new_weights_for_new_node_in_middle(self, winnerx, winnery, next_nodex, next_nodey):
        return (self.node_list[self.map[(winnerx, winnery)]] + 
                self.node_list[self.map[(next_nodex, next_nodey)]]) * 0.5

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, earlier_nodex, earlier_nodey, next_nodex, next_nodey):
        #new_node_neighbours = self._get_lattice_neighbors(next_nodex, next_nodey, 1)
        new_node_neighbours =  [(next_nodex -1, next_nodey), (next_nodex + 1, next_nodey), (next_nodex, next_nodey - 1), (next_nodex, next_nodey + 1)]
        new_node_neighbours = [n for n in new_node_neighbours if n in self.map and n != (winnerx, winnery) and n != (earlier_nodex, earlier_nodey)]
        
        # check if any other neighbour exists. If yes use that to calculate the new weights
        # e.g. Wnew = ((2*Wwinner - Wneighbour) + Wother_neighbour)/2
        if len(new_node_neighbours) > 0:
            neighbour_weights = self.node_list[[self.map[n] for n in new_node_neighbours]]
            return (2 * self.node_list[self.map[(winnerx, winnery)]] - 
                   self.node_list[self.map[(earlier_nodex, earlier_nodey)]] + 
                   neighbour_weights.sum(axis=0)) / (len(new_node_neighbours) + 1)
        else:
            # Wnew = (2*Wwinner - Wneighbour)
            return (2 * self.node_list[self.map[(winnerx, winnery)]] - 
                   self.node_list[self.map[(earlier_nodex, earlier_nodey)]])

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        winner_weights = self.node_list[self.map[(winnerx, winnery)]]
        return np.full(self.dimentions, (winner_weights.max() + winner_weights.min()) / 2)

    def grow_node(self, wx, wy, x, y, side):
        """
        grow new node if not exist on x,y coordinates using the winner node weight(wx,wy)
        check the side of the winner new node add in following order (left, right, top and bottom)
        new node N
        winner node W
        Other nodes O
        left
        =============
        1 O-N-W
        -------------
        2 N-W-O
        -------------
        3   O
            |
          N-W
        -------------
        4 N-W
            |
            O
        -------------
        =============
        right
        =============
        1 W-N-O
        -------------
        2 o-W-N
        -------------
        3 O
          |
          W-N
        -------------
        4 W-N
          |
          O
        -------------
        =============
        top
        ===============
        1 O
          |
          N
          |
          W
        -------------
        1 N
          |
          W
          |
          O
        -------------
        3 N
          |
          W-N
        -------------
        4 N
          |
        O-N
        -------------
        =============
        :param wx:
        :param wy:
        :param x:
        :param y:
        :param side:
        """
        if (x, y) in self.map:
            return
            
        # Neighbor coordinates based on side
        neighbor_map = {
            0: [(x - 1, y), (wx + 1, wy), (wx, wy + 1), (wx, wy - 1)],  # left
            1: [(x + 1, y), (wx - 1, wy), (wx, wy + 1), (wx, wy - 1)],  # right
            2: [(x, y + 1), (wx, wy - 1), (wx + 1, wy), (wx - 1, wy)],  # top
            3: [(x, y - 1), (wx, wy + 1), (wx + 1, wy), (wx - 1, wy)]   # bottom
        }
        
        coords = neighbor_map[side]
        
        if coords[0] in self.map:
            weights = self._new_weights_for_new_node_in_middle(wx, wy, coords[0][0], coords[0][1])
        elif coords[1] in self.map:
            weights = self._new_weights_for_new_node_on_one_side(wx, wy, coords[1][0], coords[1][1], x, y)
        elif coords[2] in self.map:
            weights = self._new_weights_for_new_node_on_one_side(wx, wy, coords[2][0], coords[2][1], x, y)
        elif coords[3] in self.map:
            weights = self._new_weights_for_new_node_on_one_side(wx, wy, coords[3][0], coords[3][1], x, y)
        else:
            weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
        
        np.clip(weights, 0.0, 1.0, out=weights)
        self.insert_new_node(x, y, weights)

    def _spread_error(self, x, y):
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        error = self.node_errors[self.map[(x, y)]]*self.FD ## we use FD also control the map growth
        self.node_errors[self.map[(x, y)]] = error / 2
        
        spread_error = error / 8
        for nx, ny in neighbors:
            self.node_errors[self.map[(nx, ny)]] += spread_error

    def grow_and_error_distribute(self, x, y, bmu_index):
        # This is a diamond neighborhood with radius 1
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)] # left, right, top, bottom neighbors
        
        # Check if all four neighbors exist
        if all(coord in self.map for coord in neighbors):
            self._spread_error(x, y)
        else:
            # Grow new nodes in the four sides
            for i, (nx, ny) in enumerate(neighbors):
                self.grow_node(x, y, nx, ny, i)
            self.node_errors[bmu_index] = self.groth_threshold / 2

    def winner_identification_and_weight_adaptation(self, data_sample, radius, learning_rate):
        """Batch processing version for better performance"""
        # Compute all distances at once
        distances = scipy.spatial.distance.cdist(
            self.node_list[:self.node_count], 
            data_sample.reshape(1, -1), 
            self.distance
        ).flatten()
        
        bmu_index = distances.argmin()
        error_val = distances[bmu_index]
        
        # Pre-compute Gaussian factors
        radius_sq_2 = 2.0 * radius * radius
        mask_size = round(radius)
        
        # Update neighborhood using Gaussian neighborhood function
        # SOM learning rule: wi(t+1) = wi(t) + η(t) × h(t) × (xj - wi(t))
        # where η(t) is learning_rate, h(t) is Gaussian neighborhood function
        
        bmu_x = int(self.node_coordinate[bmu_index, 0])
        bmu_y = int(self.node_coordinate[bmu_index, 1])
        
        # Update winner weights
        error = data_sample - self.node_list[bmu_index]
        self.node_list[bmu_index] += error * learning_rate
        
        # Update neighborhood            
        neighbors = self._get_lattice_neighbors(bmu_x, bmu_y, mask_size)            
        for i, j in neighbors:
            neighbor_idx = self.map[(i, j)]
            error = data_sample - self.node_list[neighbor_idx]
            #Gaussian neighborhood function h(t) = exp(-distance^2 / (2 * sigma^2)) where sigma is the current neighborhood radius
            dist_sq = (bmu_x - i)**2 + (bmu_y - j)**2
            influence = np.exp(-dist_sq / radius_sq_2)
            # Update neighbour weights using SOM weight update rule
            self.node_list[neighbor_idx] += learning_rate * influence * error
        
        return bmu_index, bmu_x, bmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        """Process data in batches for better performance"""
        for i in range(data.shape[0]):
            self.winner_identification_and_weight_adaptation(data[i], radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        """Process data in batches with growth"""
        for i in range(data.shape[0]):
            bmu_index, bmu_x, bmu_y, error_val = self.winner_identification_and_weight_adaptation(data[i], radius, learning_rate)
            
            # Handle growth for each sample in batch
            self.node_errors[bmu_index] += error_val
            if self.node_errors[bmu_index] > self.groth_threshold:
                self.grow_and_error_distribute(bmu_x, bmu_y, bmu_index)

    def fit(self, data, training_iterations, smooth_iterations, shuffle=True):
        """
        Optimized training method
        :param data: training data
        :param training_iterations: number of growing iterations
        :param smooth_iterations: number of smoothing iterations
        """
        current_learning_rate = self.learning_rate
        
        # Growing iterations
        for i in tqdm(range(training_iterations), desc="Growing"):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)

            self.grow(data, radius_exp, current_learning_rate)
            if shuffle:
                np.random.shuffle(data)
            #incrase growth threshold based on growth control factor, training iteration and current node count
            self.groth_threshold *= (1 + self.LAMBDA*(i/training_iterations)*math.log(1+self.node_count))
            
        # Smoothing iterations
        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations), desc="Smoothing"):
            radius_exp = self._get_neighbourhood_radius(smooth_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)

            self.smooth(data, radius_exp, current_learning_rate)
            if shuffle:
                np.random.shuffle(data)
        
        # Identify winners (vectorized)
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data, self.distance)
        return out.argmin(axis=0)

    def predict(self, data, index_col, label_col=None):
        """
        Optimized prediction method
        :param data: test data
        :param index_col: index column name
        :param label_col: label column name (optional)
        :return: node labels dataframe
        """
        # Prepare dataset
        weight_columns = list(data.columns.values)
        output_columns = [index_col]
        
        if label_col:
            weight_columns.remove(label_col)
            output_columns.append(label_col)
        
        weight_columns.remove(index_col)
        data_n = data[weight_columns].to_numpy()
        
        # Vectorized winner identification
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data_n, self.distance)
        winners = out.argmin(axis=0)
        
        # Build output dataframe efficiently
        data_out = data[output_columns].copy()
        data_out["output"] = winners
        
        # Group and aggregate
        grouped = data_out.groupby("output")
        
        result_data = {
            'output': [],
            index_col: [],
            'hit_count': [],
            'x': [],
            'y': []
        }
        
        if label_col:
            result_data[label_col] = []
        
        for output_id, group in grouped:
            result_data['output'].append(output_id)
            result_data[index_col].append(group[index_col].tolist())
            result_data['hit_count'].append(len(group))
            result_data['x'].append(self.node_coordinate[output_id, 0])
            result_data['y'].append(self.node_coordinate[output_id, 1])
            
            if label_col:
                result_data[label_col].append(group[label_col].tolist())
        
        self.node_labels = pd.DataFrame(result_data)
        self.output = data_out
        
        return self.node_labels


if __name__ == '__main__':
    np.random.seed(1)
    df = pd.read_csv(data_filename)
    print(df.shape)
    data_training = df.iloc[:, 1:16] #need to scale data first
    gsom = GSOM(.83, 15, max_radius=4)
    gsom.fit(data_training.to_numpy(), 100, 50)
    output = gsom.predict(df, "Name", "label")
    output.to_csv("output.csv", index=False)
    print("complete")