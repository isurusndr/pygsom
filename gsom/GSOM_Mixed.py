import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math

data_filename = "example/data/zoo.txt".replace('\\', '/')


class GSOM_Mixed:

    def __init__(self, spred_factor, num_dimensions, cat_dimensions_dict, distance='mixed', 
                 initialize='random', learning_rate=0.3, smooth_learning_factor=0.8,
                 max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=30000, random_state=42):
        """
        GSOM structure for mixed data (numerical + categorical)
        
        :param spred_factor: spread factor of GSOM graph
        :param num_dimensions: number of numerical features
        :param cat_dimensions_dict: dictionary {feature_index: num_categories} e.g., {16: 7, 17: 2}
        :param distance: distance method (use 'mixed' for mixed data)
        :param initialize: weight vector initialize method
        :param learning_rate: initial training learning rate
        :param smooth_learning_factor: smooth learning factor
        :param max_radius: maximum neighbourhood radius
        :param FD: spread weight value
        :param r: learning rate update value
        :param alpha: learning rate update value
        :param initial_node_size: initial node allocation in memory
        :param random_state: random seed for reproducibility
        """
        self.initial_node_size = initial_node_size
        self.node_count = 0
        self.map = {}
        
        # Numerical weight storage
        self.num_dimensions = num_dimensions
        self.node_list_num = np.zeros((self.initial_node_size, num_dimensions))
        
        # Categorical probability storage
        self.cat_dimensions_dict = cat_dimensions_dict
        self.cat_features = sorted(cat_dimensions_dict.keys())
        self.node_list_cat = [{} for _ in range(self.initial_node_size)]
        
        self.node_coordinate = np.zeros((self.initial_node_size, 2), dtype=np.int32)
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.float64)
        
        # Calculate total dimensionality for growth threshold
        total_dim = num_dimensions + len(cat_dimensions_dict)
        self.spred_factor = spred_factor
        self.groth_threshold = -total_dim * math.log(self.spred_factor)
        
        self.FD = FD
        self.R = r
        self.ALPHA = alpha
        self.LAMBDA = np.clip(1 - self.spred_factor, 0.1, 0.9)
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
        """Initialize 4 corner nodes"""
        np.random.seed(self.random_state)
        self.insert_node_with_weights(1, 1)
        self.insert_node_with_weights(1, 0)
        self.insert_node_with_weights(0, 1)
        self.insert_node_with_weights(0, 0)

    def insert_new_node(self, x, y, num_weights, cat_probs):
        """Insert a new node with both numerical weights and categorical probabilities"""
        if self.node_count >= self.initial_node_size:
            # Resize arrays dynamically
            new_size = self.initial_node_size * 2
            self.node_list_num = np.resize(self.node_list_num, (new_size, self.num_dimensions))
            self.node_coordinate = np.resize(self.node_coordinate, (new_size, 2))
            self.node_errors = np.resize(self.node_errors, new_size)
            
            # Extend categorical storage
            self.node_list_cat.extend([{} for _ in range(new_size - self.initial_node_size)])
            self.initial_node_size = new_size
            
        self.map[(x, y)] = self.node_count
        self.node_list_num[self.node_count] = num_weights
        self.node_list_cat[self.node_count] = cat_probs
        self.node_coordinate[self.node_count] = [x, y]
        self.node_count += 1

    def insert_node_with_weights(self, x, y):
        """Initialize node weights randomly"""
        if self.initialize == 'random':
            # Initialize numerical weights
            num_weights = np.random.rand(self.num_dimensions)
            
            # Initialize categorical probabilities uniformly
            cat_probs = {}
            for feat_idx, num_cats in self.cat_dimensions_dict.items():
                cat_probs[feat_idx] = np.ones(num_cats) / num_cats
        else:
            print("initialize method not support")
            return
            
        self.insert_new_node(x, y, num_weights, cat_probs)

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(- iteration / time_constant)
    
    def _get_lattice_neighbors(self, x, y, radius):
        """Optimized neighbor search with diamond neighbourhood"""
        neighbors = []
        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if (i, j) in self.map and (i, j) != (x, y):
                    if abs(i - x) + abs(j - y) <= radius:
                        neighbors.append((i, j))
        return neighbors

    def _new_weights_for_new_node_in_middle(self, winnerx, winnery, next_nodex, next_nodey):
        """Calculate weights for new node between two existing nodes"""
        winner_idx = self.map[(winnerx, winnery)]
        neighbor_idx = self.map[(next_nodex, next_nodey)]
        
        # Numerical weights - average
        num_weights = (self.node_list_num[winner_idx] + self.node_list_num[neighbor_idx]) * 0.5
        
        # Categorical probabilities - average and normalize
        cat_probs = {}
        for feat_idx in self.cat_features:
            prob_avg = (self.node_list_cat[winner_idx][feat_idx] + 
                       self.node_list_cat[neighbor_idx][feat_idx]) * 0.5
            cat_probs[feat_idx] = prob_avg / prob_avg.sum()
            
        return num_weights, cat_probs

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, earlier_nodex, earlier_nodey, next_nodex, next_nodey):
        """Calculate weights for new node on one side with improved logic"""
        winner_idx = self.map[(winnerx, winnery)]
        earlier_idx = self.map[(earlier_nodex, earlier_nodey)]
        
        # Find other neighbors of the new node position
        new_node_neighbours = [(next_nodex - 1, next_nodey), (next_nodex + 1, next_nodey), 
                              (next_nodex, next_nodey - 1), (next_nodex, next_nodey + 1)]
        new_node_neighbours = [n for n in new_node_neighbours if n in self.map 
                              and n != (winnerx, winnery) and n != (earlier_nodex, earlier_nodey)]
        
        if len(new_node_neighbours) > 0:
            # Use other neighbors for better interpolation
            # Numerical weights
            neighbour_indices = [self.map[n] for n in new_node_neighbours]
            neighbour_weights = self.node_list_num[neighbour_indices]
            num_weights = (2 * self.node_list_num[winner_idx] - 
                          self.node_list_num[earlier_idx] + 
                          neighbour_weights.sum(axis=0)) / (len(new_node_neighbours) + 1)
            
            # Categorical probabilities
            cat_probs = {}
            for feat_idx in self.cat_features:
                prob_sum = sum([self.node_list_cat[idx][feat_idx] for idx in neighbour_indices])
                prob_extrap = (2 * self.node_list_cat[winner_idx][feat_idx] - 
                              self.node_list_cat[earlier_idx][feat_idx] + 
                              prob_sum) / (len(new_node_neighbours) + 1)
                prob_extrap = np.maximum(prob_extrap, 0.0001)
                cat_probs[feat_idx] = prob_extrap / prob_extrap.sum()
        else:
            # Simple extrapolation
            num_weights = (2 * self.node_list_num[winner_idx] - self.node_list_num[earlier_idx])
            
            cat_probs = {}
            for feat_idx in self.cat_features:
                prob_extrap = (2 * self.node_list_cat[winner_idx][feat_idx] - 
                              self.node_list_cat[earlier_idx][feat_idx])
                prob_extrap = np.maximum(prob_extrap, 0.0001)
                cat_probs[feat_idx] = prob_extrap / prob_extrap.sum()
        
        return num_weights, cat_probs

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        """Calculate weights when only one neighbor exists"""
        winner_idx = self.map[(winnerx, winnery)]
        
        # Numerical weights
        winner_weights = self.node_list_num[winner_idx]
        num_weights = np.full(self.num_dimensions, (winner_weights.max() + winner_weights.min()) / 2)
        
        # Categorical probabilities - copy from winner
        cat_probs = {}
        for feat_idx in self.cat_features:
            cat_probs[feat_idx] = self.node_list_cat[winner_idx][feat_idx].copy()
            
        return num_weights, cat_probs

    def compute_mixed_distance(self, data_point, node_idx):
        """
        Compute mixed distance: D_total = D_num + D_cat
        
        D_num = Euclidean distance for numerical features
        D_cat = sum of (1 - P(category))^2 for categorical features
        """
        # Numerical distance
        num_data = data_point[:self.num_dimensions]
        d_num = np.linalg.norm(num_data - self.node_list_num[node_idx])
        
        # Categorical distance
        d_cat = 0.0
        for feat_idx in self.cat_features:
            category = int(data_point[feat_idx])
            prob = self.node_list_cat[node_idx][feat_idx][category]
            d_cat += (1 - prob) ** 2
            
        return d_num + d_cat

    def compute_mixed_distance_batch(self, data_point):
        """Optimized batch distance computation for all nodes"""
        # Numerical distance for all nodes
        num_data = data_point[:self.num_dimensions]
        d_num = np.linalg.norm(self.node_list_num[:self.node_count] - num_data, axis=1)
        
        # Categorical distance for all nodes
        d_cat = np.zeros(self.node_count)
        for feat_idx in self.cat_features:
            category = int(data_point[feat_idx])
            for node_idx in range(self.node_count):
                prob = self.node_list_cat[node_idx][feat_idx][category]
                d_cat[node_idx] += (1 - prob) ** 2
        
        return d_num + d_cat

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
            num_weights, cat_probs = self._new_weights_for_new_node_in_middle(wx, wy, coords[0][0], coords[0][1])
        elif coords[1] in self.map:
            num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, coords[1][0], coords[1][1], x, y)
        elif coords[2] in self.map:
            num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, coords[2][0], coords[2][1], x, y)
        elif coords[3] in self.map:
            num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, coords[3][0], coords[3][1], x, y)
        else:
            num_weights, cat_probs = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
        
        np.clip(num_weights, 0.0, 1.0, out=num_weights)
        self.insert_new_node(x, y, num_weights, cat_probs)

    def _spread_error(self, x, y):
        """Distribute error to neighbors"""
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        error = self.node_errors[self.map[(x, y)]] * self.FD
        self.node_errors[self.map[(x, y)]] = error / 2
        
        spread_error = error / 8
        for nx, ny in neighbors:
            self.node_errors[self.map[(nx, ny)]] += spread_error

    def grow_and_error_distribute(self, x, y, bmu_index):
        """Check for growth and distribute error"""
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        
        if all(coord in self.map for coord in neighbors):
            self._spread_error(x, y)
        else:
            for i, (nx, ny) in enumerate(neighbors):
                self.grow_node(x, y, nx, ny, i)
            self.node_errors[bmu_index] = self.groth_threshold / 2

    def winner_identification_and_weight_adaptation(self, data_sample, radius, learning_rate):
        """Optimized winner identification and weight adaptation for mixed data"""
        # Compute all distances at once
        distances = self.compute_mixed_distance_batch(data_sample)
        
        bmu_index = distances.argmin()
        error_val = distances[bmu_index]
        
        bmu_x = int(self.node_coordinate[bmu_index, 0])
        bmu_y = int(self.node_coordinate[bmu_index, 1])
        
        # Update winner weights
        # Numerical features
        num_data = data_sample[:self.num_dimensions]
        num_error = num_data - self.node_list_num[bmu_index]
        self.node_list_num[bmu_index] += num_error * learning_rate        
        # Categorical features - update probabilities of winner
        for feat_idx in self.cat_features:
            category = int(data_sample[feat_idx])
            self.node_list_cat[bmu_index][feat_idx] *= (1 - learning_rate) ## TODO: here we adjust the probabilities for all category values first
            self.node_list_cat[bmu_index][feat_idx][category] += learning_rate ## TODO: and then increase the probability for the actual category
            self.node_list_cat[bmu_index][feat_idx] /= self.node_list_cat[bmu_index][feat_idx].sum() #then normalize so that sum of probabilities is 1
        
        # Update neighborhood
        radius_sq_2 = 2.0 * radius * radius
        mask_size = round(radius)
        neighbors = self._get_lattice_neighbors(bmu_x, bmu_y, mask_size)
        
        for i, j in neighbors:
            neighbor_idx = self.map[(i, j)]
            
            # Gaussian neighborhood function
            dist_sq = (bmu_x - i)**2 + (bmu_y - j)**2
            influence = np.exp(-dist_sq / radius_sq_2)
            
            # Update numerical weights
            num_error = num_data - self.node_list_num[neighbor_idx]
            self.node_list_num[neighbor_idx] += learning_rate * influence * num_error
            
            # Update categorical probabilities
            for feat_idx in self.cat_features:
                category = int(data_sample[feat_idx])
                self.node_list_cat[neighbor_idx][feat_idx] *= (1 - learning_rate * influence)
                self.node_list_cat[neighbor_idx][feat_idx][category] += learning_rate * influence
                self.node_list_cat[neighbor_idx][feat_idx] /= \
                    self.node_list_cat[neighbor_idx][feat_idx].sum()
        
        return bmu_index, bmu_x, bmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        """Smoothing phase"""
        for i in range(data.shape[0]):
            self.winner_identification_and_weight_adaptation(data[i], radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        """Growing phase"""
        for i in range(data.shape[0]):
            bmu_index, bmu_x, bmu_y, error_val = \
                self.winner_identification_and_weight_adaptation(data[i], radius, learning_rate)
            
            self.node_errors[bmu_index] += error_val
            if self.node_errors[bmu_index] > self.groth_threshold:
                self.grow_and_error_distribute(bmu_x, bmu_y, bmu_index)

    def fit(self, data, training_iterations, smooth_iterations, shuffle=True):
        """
        Optimized training method
        :param data: training data
        :param training_iterations: number of growing iterations
        :param smooth_iterations: number of smoothing iterations
        :param shuffle: whether to shuffle data between iterations
        """
        current_learning_rate = self.learning_rate
        
        # Growing iterations
        for i in tqdm(range(training_iterations), desc="Growing"):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
                #incrase growth threshold based on growth control factor, training iteration and current node count
                decay = np.clip(1-(i/training_iterations), 0.1, 0.9) # progressive decay factor to prevent early over growth
                self.groth_threshold *= (1 + self.LAMBDA*decay*math.log(1+self.node_count))

            self.grow(data, radius_exp, current_learning_rate)
            if shuffle:
                np.random.shuffle(data)
            
        # Smoothing iterations
        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations), desc="Smoothing"):
            radius_exp = self._get_neighbourhood_radius(smooth_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)

            self.smooth(data, radius_exp, current_learning_rate)
            if shuffle:
                np.random.shuffle(data)
        
        # Identify winners
        winners = []
        for data_idx in range(data.shape[0]):
            distances = self.compute_mixed_distance_batch(data[data_idx])
            winners.append(distances.argmin())
        
        return np.array(winners)

    def predict(self, data, index_col, label_col=None):
        """
        Optimized prediction method for mixed data
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
        winners = []
        for data_idx in range(data_n.shape[0]):
            distances = self.compute_mixed_distance_batch(data_n[data_idx])
            winners.append(distances.argmin())
        
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
    print(f"Data shape: {df.shape}")
    
    # There are 15 boolean features corresponding to the presence of hair, feathers, eggs, milk, 
    # backbone, ﬁns, tail; and whether airborne, aquatic, predator, toothed, breathes, venomous, domestic, catsize. 
    # The numeric attribute corresponds to the number of legs
    num_features = 1
    cat_features_dict = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 
                         9: 2, 10: 2, 11: 2, 13: 2, 14: 2, 15: 2}  # {feature_idx: num_categories}
    
    data_training = df.iloc[:, 1:16].to_numpy()
    
    gsom = GSOM_Mixed(
        spred_factor=0.83,
        num_dimensions=num_features,
        cat_dimensions_dict=cat_features_dict,
        max_radius=4
    )
    
    gsom.fit(data_training, 100, 50, shuffle=True)
    output = gsom.predict(df, "Name", "label")
    output.to_csv("output_mixed.csv", index=False)
    
    print(f"Training complete. Total nodes: {gsom.node_count}")