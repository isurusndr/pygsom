import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math

data_filename = "example/data/zoo.txt".replace('\\', '/')


class GSOM_Mixed:

    def __init__(self, spred_factor, num_dimensions, cat_dimensions, distance='mixed', 
                 initialize='random', learning_rate=0.3, smooth_learning_factor=0.8,
                 max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=30000, random_state=42):
        """
        GSOM for mixed data (numerical + categorical)
        :param spred_factor: spread factor of GSOM graph
        :param num_dimensions: number of numerical features
        :param cat_dimensions: dict mapping categorical feature indices to number of categories
                              e.g., {16: 7, 17: 5} means feature 16 has 7 categories, feature 17 has 5
        :param distance: distance method (use 'mixed' for FMSOM approach)
        :param initialize: weight vector initialize method
        :param learning_rate: initial training learning rate
        :param smooth_learning_factor: smooth learning factor
        :param max_radius: maximum neighbourhood radius
        :param FD: spread weight value
        :param r: learning rate update value
        :param alpha: learning rate update value
        :param initial_node_size: initial node allocation in memory
        """
        self.initial_node_size = initial_node_size
        self.node_count = 0
        self.map = {}
        
        # Separate storage for numerical and categorical weights
        self.num_dimensions = num_dimensions
        self.cat_dimensions = cat_dimensions  # dict: {feature_idx: num_categories}
        self.total_dimensions = num_dimensions + len(cat_dimensions)
        
        # Numerical weights
        self.node_list_num = np.zeros((self.initial_node_size, num_dimensions))
        
        # Categorical probability tables: dict of {feature_idx: array[node_count, num_categories]}
        self.node_list_cat = {}
        for feat_idx, num_cats in cat_dimensions.items():
            self.node_list_cat[feat_idx] = np.zeros((self.initial_node_size, num_cats))
        
        self.node_coordinate = np.zeros((self.initial_node_size, 2), dtype=np.int32)
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.float64)
        
        self.spred_factor = spred_factor
        self.groth_threshold = -self.total_dimensions * math.log(self.spred_factor)
        self.FD = FD
        self.R = r
        self.ALPHA = alpha
        self.distance = distance
        self.initialize = initialize
        self.learning_rate = learning_rate
        self.smooth_learning_factor = smooth_learning_factor
        self.max_radius = max_radius
        self.random_state = random_state
        self.initialize_GSOM()
        self.node_labels = None
        self.output = None

    def initialize_GSOM(self):
        np.random.seed(self.random_state)
        self.insert_node_with_weights(1, 1)
        self.insert_node_with_weights(1, 0)
        self.insert_node_with_weights(0, 1)
        self.insert_node_with_weights(0, 0)

    def insert_new_node(self, x, y, num_weights, cat_weights):
        if self.node_count >= self.initial_node_size:
            # Resize arrays dynamically
            new_size = self.initial_node_size * 2
            self.node_list_num = np.resize(self.node_list_num, (new_size, self.num_dimensions))
            for feat_idx in self.cat_dimensions:
                self.node_list_cat[feat_idx] = np.resize(
                    self.node_list_cat[feat_idx], 
                    (new_size, self.cat_dimensions[feat_idx])
                )
            self.node_coordinate = np.resize(self.node_coordinate, (new_size, 2))
            self.node_errors = np.resize(self.node_errors, new_size)
            self.initial_node_size = new_size
            
        self.map[(x, y)] = self.node_count
        self.node_list_num[self.node_count] = num_weights
        for feat_idx, probs in cat_weights.items():
            self.node_list_cat[feat_idx][self.node_count] = probs
        self.node_coordinate[self.node_count] = [x, y]
        self.node_count += 1

    def insert_node_with_weights(self, x, y):
        if self.initialize == 'random':
            # Initialize numerical weights randomly
            num_weights = np.random.rand(self.num_dimensions)
            
            # Initialize categorical probabilities uniformly
            cat_weights = {}
            for feat_idx, num_cats in self.cat_dimensions.items():
                cat_weights[feat_idx] = np.ones(num_cats) / num_cats
        else:
            print("initialize method not support")
            
        self.insert_new_node(x, y, num_weights, cat_weights)

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(- iteration / time_constant)
    
    def _get_lattice_neighbors(self, x, y, radius):
        """Get neighbors within radius using Manhattan distance"""
        neighbors = []
        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if (i, j) in self.map and (i, j) != (x, y):
                    if abs(i - x) + abs(j - y) <= radius:
                        neighbors.append((i, j))
        return neighbors

    def compute_mixed_distance(self, data_sample, node_idx):
        """
        Compute mixed distance: D_total = D_num + D_cat
        """
        # Numerical distance (Euclidean)
        num_data = data_sample[:self.num_dimensions]
        D_num = np.sqrt(np.sum((num_data - self.node_list_num[node_idx])**2))
        
        # Categorical distance (probability-based)
        D_cat = 0
        cat_start_idx = self.num_dimensions
        for i, feat_idx in enumerate(sorted(self.cat_dimensions.keys())):
            cat_value = int(data_sample[cat_start_idx + i])
            prob = self.node_list_cat[feat_idx][node_idx, cat_value]
            D_cat += (1 - prob)**2
        
        return D_num + D_cat

    def _new_weights_for_new_node_in_middle(self, winnerx, winnery, next_nodex, next_nodey):
        """Average weights from two existing nodes"""
        num_weights = (self.node_list_num[self.map[(winnerx, winnery)]] + 
                      self.node_list_num[self.map[(next_nodex, next_nodey)]]) * 0.5
        
        cat_weights = {}
        for feat_idx in self.cat_dimensions:
            cat_weights[feat_idx] = (self.node_list_cat[feat_idx][self.map[(winnerx, winnery)]] + 
                                     self.node_list_cat[feat_idx][self.map[(next_nodex, next_nodey)]]) * 0.5
        
        return num_weights, cat_weights

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, earlier_nodex, earlier_nodey, 
                                               next_nodex, next_nodey):
        """Calculate weights for new node with one older neighbor"""
        new_node_neighbours = [(next_nodex - 1, next_nodey), (next_nodex + 1, next_nodey), 
                               (next_nodex, next_nodey - 1), (next_nodex, next_nodey + 1)]
        new_node_neighbours = [n for n in new_node_neighbours if n in self.map and 
                              n != (winnerx, winnery) and n != (earlier_nodex, earlier_nodey)]
        
        if len(new_node_neighbours) > 0:
            # Numerical weights
            neighbour_num_weights = self.node_list_num[[self.map[n] for n in new_node_neighbours]]
            num_weights = (2 * self.node_list_num[self.map[(winnerx, winnery)]] - 
                          self.node_list_num[self.map[(earlier_nodex, earlier_nodey)]] + 
                          neighbour_num_weights.sum(axis=0)) / (len(new_node_neighbours) + 1)
            
            # Categorical weights
            cat_weights = {}
            for feat_idx in self.cat_dimensions:
                neighbour_cat_weights = self.node_list_cat[feat_idx][[self.map[n] for n in new_node_neighbours]]
                cat_weights[feat_idx] = (2 * self.node_list_cat[feat_idx][self.map[(winnerx, winnery)]] - 
                                        self.node_list_cat[feat_idx][self.map[(earlier_nodex, earlier_nodey)]] + 
                                        neighbour_cat_weights.sum(axis=0)) / (len(new_node_neighbours) + 1)
        else:
            # Numerical weights
            num_weights = (2 * self.node_list_num[self.map[(winnerx, winnery)]] - 
                          self.node_list_num[self.map[(earlier_nodex, earlier_nodey)]])
            
            # Categorical weights
            cat_weights = {}
            for feat_idx in self.cat_dimensions:
                cat_weights[feat_idx] = (2 * self.node_list_cat[feat_idx][self.map[(winnerx, winnery)]] - 
                                        self.node_list_cat[feat_idx][self.map[(earlier_nodex, earlier_nodey)]])
        
        return num_weights, cat_weights

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        """Initialize weights for new node with only one neighbor"""
        # Numerical weights
        winner_num_weights = self.node_list_num[self.map[(winnerx, winnery)]]
        num_weights = np.full(self.num_dimensions, (winner_num_weights.max() + winner_num_weights.min()) / 2)
        
        # Categorical weights - uniform distribution
        cat_weights = {}
        for feat_idx, num_cats in self.cat_dimensions.items():
            cat_weights[feat_idx] = np.ones(num_cats) / num_cats
        
        return num_weights, cat_weights

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
            num_weights, cat_weights = self._new_weights_for_new_node_in_middle(
                wx, wy, coords[0][0], coords[0][1])
        elif coords[1] in self.map:
            num_weights, cat_weights = self._new_weights_for_new_node_on_one_side(
                wx, wy, coords[1][0], coords[1][1], x, y)
        elif coords[2] in self.map:
            num_weights, cat_weights = self._new_weights_for_new_node_on_one_side(
                wx, wy, coords[2][0], coords[2][1], x, y)
        elif coords[3] in self.map:
            num_weights, cat_weights = self._new_weights_for_new_node_on_one_side(
                wx, wy, coords[3][0], coords[3][1], x, y)
        else:
            num_weights, cat_weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
        
        # Clip numerical weights
        np.clip(num_weights, 0.0, 1.0, out=num_weights)
        
        # Normalize categorical probabilities
        for feat_idx in cat_weights:
            cat_weights[feat_idx] = np.clip(cat_weights[feat_idx], 0.0, 1.0)
            prob_sum = cat_weights[feat_idx].sum()
            if prob_sum > 0:
                cat_weights[feat_idx] /= prob_sum
            else:
                cat_weights[feat_idx] = np.ones(self.cat_dimensions[feat_idx]) / self.cat_dimensions[feat_idx]
        
        self.insert_new_node(x, y, num_weights, cat_weights)

    def _spread_error(self, x, y):
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        error = self.node_errors[self.map[(x, y)]] * self.FD
        self.node_errors[self.map[(x, y)]] = error / 2
        
        spread_error = error / 8
        for nx, ny in neighbors:
            self.node_errors[self.map[(nx, ny)]] += spread_error

    def grow_and_error_distribute(self, x, y, bmu_index):
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        
        if all(coord in self.map for coord in neighbors):
            self._spread_error(x, y)
        else:
            for i, (nx, ny) in enumerate(neighbors):
                self.grow_node(x, y, nx, ny, i)
            self.node_errors[bmu_index] = self.groth_threshold / 2

    def winner_identification_and_weight_adaptation(self, data_sample, radius, learning_rate):
        """
        FMSOM-style weight adaptation for mixed data
        Process a single sample with batch-style accumulation
        """
        # 1. COMPETITION: Find BMU using mixed distance
        distances = np.array([self.compute_mixed_distance(data_sample, i) 
                             for i in range(self.node_count)])
        bmu_index = distances.argmin()
        error_val = distances[bmu_index]
        
        bmu_x = int(self.node_coordinate[bmu_index, 0])
        bmu_y = int(self.node_coordinate[bmu_index, 1])
        
        # 2. COOPERATION: Determine neighborhood
        radius_sq_2 = 2.0 * radius * radius
        mask_size = round(radius)
        neighbors = self._get_lattice_neighbors(bmu_x, bmu_y, mask_size)
        #### neighbors.append((bmu_x, bmu_y))  # Include BMU itself (to update winner weights)
        
        # Initialize accumulators
        accum_neigh = {}
        num_update = {}
        cat_update = {}
        
        num_data = data_sample[:self.num_dimensions]
        cat_data = data_sample[self.num_dimensions:]
        
        for i, j in neighbors:
            neighbor_idx = self.map[(i, j)]
            
            # Compute Gaussian influence
            dist_sq = (bmu_x - i)**2 + (bmu_y - j)**2
            h = np.exp(-dist_sq / radius_sq_2)
            
            accum_neigh[neighbor_idx] = h
            
            # Numerical contribution
            num_update[neighbor_idx] = h * num_data
            
            # Categorical contribution
            cat_update[neighbor_idx] = {}
            for cat_idx, feat_idx in enumerate(sorted(self.cat_dimensions.keys())):
                cat_value = int(cat_data[cat_idx])
                cat_update[neighbor_idx][feat_idx] = {cat_value: h}
        
        # 3. ADAPTATION: Update prototypes
        for neighbor_idx in accum_neigh:
            h_sum = accum_neigh[neighbor_idx]
            if h_sum > 0:
                # Update numerical weights
                self.node_list_num[neighbor_idx] += learning_rate * (
                    num_update[neighbor_idx] / h_sum - self.node_list_num[neighbor_idx]
                )
                
                # Update categorical probabilities
                for feat_idx in self.cat_dimensions:
                    if feat_idx in cat_update[neighbor_idx]:
                        for cat_value, h_val in cat_update[neighbor_idx][feat_idx].items():
                            # Smooth update towards observed category
                            target_probs = np.zeros(self.cat_dimensions[feat_idx])
                            target_probs[cat_value] = 1.0
                            
                            self.node_list_cat[feat_idx][neighbor_idx] += learning_rate * (
                                h_val / h_sum * target_probs - self.node_list_cat[feat_idx][neighbor_idx]
                            )
                            
                            # Normalize probabilities
                            prob_sum = self.node_list_cat[feat_idx][neighbor_idx].sum()
                            if prob_sum > 0:
                                self.node_list_cat[feat_idx][neighbor_idx] /= prob_sum
        
        return bmu_index, bmu_x, bmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        """Process data one sample at a time for smoothing"""
        for i in range(data.shape[0]):
            self.winner_identification_and_weight_adaptation(data[i], radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        """Process data one sample at a time with growth"""
        for i in range(data.shape[0]):
            bmu_index, bmu_x, bmu_y, error_val = self.winner_identification_and_weight_adaptation(
                data[i], radius, learning_rate
            )
            
            self.node_errors[bmu_index] += error_val
            if self.node_errors[bmu_index] > self.groth_threshold:
                self.grow_and_error_distribute(bmu_x, bmu_y, bmu_index)

    def fit(self, data, training_iterations, smooth_iterations, shuffle=True):
        """
        Training method for mixed data
        :param data: training data (numerical features followed by categorical)
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

            self.grow(data, radius_exp, current_learning_rate)
            if shuffle:
                np.random.shuffle(data)
            
            #incrase growth threshold based on spred_factor and learning rate and iteration
            self.groth_threshold *= (1 + math.log(self.node_count) * (1 - self.spred_factor) * 
                                    (1 - (i / training_iterations)))
            
        # Smoothing iterations
        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations), desc="Smoothing"):
            radius_exp = self._get_neighbourhood_radius(smooth_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)

            self.smooth(data, radius_exp, current_learning_rate)
            if shuffle:
                np.random.shuffle(data)
        
        # Identify final winners
        winners = []
        for sample in data:
            distances = np.array([self.compute_mixed_distance(sample, i) 
                                 for i in range(self.node_count)])
            winners.append(distances.argmin())
        
        return np.array(winners)

    def predict(self, data, index_col, label_col=None, cat_feature_names=None):
        """
        Prediction method for mixed data
        :param data: test data (DataFrame)
        :param index_col: index column name
        :param label_col: label column name (optional)
        :param cat_feature_names: list of categorical feature column names
        :return: node labels dataframe
        """
        # Prepare dataset
        weight_columns = list(data.columns.values)
        output_columns = [index_col]
        
        if label_col:
            weight_columns.remove(label_col)
            output_columns.append(label_col)
        
        weight_columns.remove(index_col)
        
        # Separate numerical and categorical features
        if cat_feature_names:
            num_columns = [col for col in weight_columns if col not in cat_feature_names]
            cat_columns = cat_feature_names
        else:
            num_columns = weight_columns[:self.num_dimensions]
            cat_columns = weight_columns[self.num_dimensions:]
        
        # Create combined data array
        data_n = np.hstack([
            data[num_columns].to_numpy(),
            data[cat_columns].to_numpy()
        ])
        
        # Find winners
        winners = []
        for sample in data_n:
            distances = np.array([self.compute_mixed_distance(sample, i) 
                                 for i in range(self.node_count)])
            winners.append(distances.argmin())
        
        # Build output dataframe
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
    
    # Example: First 16 features are numerical, feature 16 is categorical with 7 categories
    data_training_num = df.iloc[:, 1:17].to_numpy()
    
    # If you have categorical features, specify them
    # For example: if column 17 (index 16 in 0-indexed) has 7 categories
    cat_dims = {16: 7}  # feature index 16 has 7 categories
    
    # Create GSOM for mixed data
    gsom = GSOM_Mixed(.83, 16, cat_dims, max_radius=4)
    
    # Prepare training data (numerical + categorical)
    # If you have categorical data, concatenate it
    # data_training = np.hstack([data_training_num, categorical_data])
    data_training = data_training_num  # For now, using only numerical
    
    gsom.fit(data_training, 100, 50)
    output = gsom.predict(df, "Name", "label")
    output.to_csv("output_mixed.csv", index=False)
    print("complete")