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
                 max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=30000):
        """
        GSOM structure for mixed data (numerical + categorical):
        Numerical features handled with Euclidean distance, and categorical features handled with frequency neurons
        Reference: 
        Del Coso, Carmelo, et al. "Mixing numerical and categorical data in a self-organizing map by means of frequency neurons." Applied Soft Computing 36 (2015): 246-254.
        
        :param spred_factor: spread factor of GSOM graph
        :param num_dimensions: number of numerical features
        :param cat_dimensions_dict: dictionary {feature_index: num_categories} for categorical features
                                    e.g., {16: 7, 17: 2} means feature 16 has 7 categories, feature 17 has 2
        :param distance: distance method (use 'mixed' for mixed data)
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
        
        # Numerical weight storage
        self.num_dimensions = num_dimensions
        self.node_list_num = np.zeros((self.initial_node_size, num_dimensions))
        
        # Categorical probability storage
        self.cat_dimensions_dict = cat_dimensions_dict  # {feature_idx: num_categories}
        self.cat_features = sorted(cat_dimensions_dict.keys())
        # Store probability tables: node_list_cat[node_idx][feature_idx] = prob_array
        self.node_list_cat = [{} for _ in range(self.initial_node_size)]
        
        self.node_coordinate = np.zeros((self.initial_node_size, 2))
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.longdouble)
        
        # Calculate total dimensionality for growth threshold
        total_dim = num_dimensions + len(cat_dimensions_dict)
        self.spred_factor = spred_factor
        self.groth_threshold = -total_dim * math.log(self.spred_factor)
        
        self.FD = FD
        self.R = r
        self.ALPHA = alpha
        self.distance = distance
        self.initialize = initialize
        self.learning_rate = learning_rate
        self.smooth_learning_factor = smooth_learning_factor
        self.max_radius = max_radius
        
        self.initialize_GSOM()
        self.node_labels = None
        self.output = None

    def initialize_GSOM(self):
        """Initialize 4 corner nodes"""
        self._insert_node_with_weights(1, 1)
        self._insert_node_with_weights(1, 0)
        self._insert_node_with_weights(0, 1)
        self._insert_node_with_weights(0, 0)

    def _insert_new_node(self, x, y, num_weights, cat_probs):
        """Insert a new node with both numerical weights and categorical probabilities"""
        if self.node_count >= self.initial_node_size:
            print("node size out of bound")
            return
            
        self.map[(x, y)] = self.node_count
        self.node_list_num[self.node_count] = num_weights
        self.node_list_cat[self.node_count] = cat_probs
        self.node_coordinate[self.node_count][0] = x
        self.node_coordinate[self.node_count][1] = y
        self.node_count += 1

    def _insert_node_with_weights(self, x, y):
        """Initialize node weights randomly"""
        if self.initialize == 'random':
            # Initialize numerical weights
            num_weights = np.random.rand(self.num_dimensions)
            
            # Initialize categorical probabilities uniformly
            cat_probs = {}
            for feat_idx, num_cats in self.cat_dimensions_dict.items():
                cat_probs[feat_idx] = np.ones(num_cats) / num_cats
        else:
            print("initialize method not supported")
            return
            
        self._insert_new_node(x, y, num_weights, cat_probs)

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(- iteration / time_constant)

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
            # Normalize to ensure sum = 1
            cat_probs[feat_idx] = prob_avg / prob_avg.sum()
            
        return num_weights, cat_probs

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, next_nodex, next_nodey):
        """Calculate weights for new node on one side"""
        winner_idx = self.map[(winnerx, winnery)]
        neighbor_idx = self.map[(next_nodex, next_nodey)]
        
        # Numerical weights - extrapolate
        num_weights = (2 * self.node_list_num[winner_idx] - self.node_list_num[neighbor_idx])
        
        # Categorical probabilities - extrapolate and normalize
        cat_probs = {}
        for feat_idx in self.cat_features:
            prob_extrap = (2 * self.node_list_cat[winner_idx][feat_idx] - 
                          self.node_list_cat[neighbor_idx][feat_idx])
            # Clip negative values
            prob_extrap = np.maximum(prob_extrap, 0.0001)
            # Normalize
            cat_probs[feat_idx] = prob_extrap / prob_extrap.sum()
            
        return num_weights, cat_probs

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        """Calculate weights when only one neighbor exists"""
        winner_idx = self.map[(winnerx, winnery)]
        
        # Numerical weights
        num_weights = np.full(self.num_dimensions, 
                             (self.node_list_num[winner_idx].max() + 
                              self.node_list_num[winner_idx].min()) / 2)
        
        # Categorical probabilities - copy from winner
        cat_probs = {}
        for feat_idx in self.cat_features:
            cat_probs[feat_idx] = self.node_list_cat[winner_idx][feat_idx].copy()
            
        return num_weights, cat_probs

    def _compute_mixed_distance(self, data_point, node_idx):
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

    def _grow_node(self, wx, wy, x, y, side):
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
            
        if side == 0:  # add new node to left of winner
            if (x - 1, y) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_in_middle(wx, wy, x - 1, y)
            elif (wx + 1, wy) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
            elif (wx, wy + 1) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
            elif (wx, wy - 1) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
            else:
                num_weights, cat_probs = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
                
        elif side == 1:  # add new node to right of winner
            if (x + 1, y) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_in_middle(wx, wy, x + 1, y)
            elif (wx - 1, wy) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
            elif (wx, wy + 1) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
            elif (wx, wy - 1) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
            else:
                num_weights, cat_probs = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
                
        elif side == 2:  # add new node to top of winner
            if (x, y + 1) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_in_middle(wx, wy, x, y + 1)
            elif (wx, wy - 1) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
            elif (wx + 1, wy) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
            elif (wx - 1, wy) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
            else:
                num_weights, cat_probs = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
                
        else:  # side == 3, bottom
            if (x, y - 1) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_in_middle(wx, wy, x, y - 1)
            elif (wx, wy + 1) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
            elif (wx + 1, wy) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
            elif (wx - 1, wy) in self.map:
                num_weights, cat_probs = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
            else:
                num_weights, cat_probs = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
        
        # Clip numerical weights between (0,1)
        num_weights = np.clip(num_weights, 0.0, 1.0)        
        self._insert_new_node(x, y, num_weights, cat_probs)

    def gussian_neighbourhood_function(self, distance, sigma):
        """Gaussian neighborhood function h(t) = exp(-distance^2 / (2 * sigma^2))"""
        return np.exp(-1.0 * distance / (2.0 * (sigma * sigma)))
    
    def lattice_distance(self, coord1, coord2):
        """Euclidean distance between two coordinates in the lattice"""
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)  
    
    def get_lattice_neighbors(self, bmu_coord, radius):
        """Get circular neighbourhood mask"""
        neighbors = []
        # Iterate over the winner node radius(neighbourhood) in the lattice
        for i in range(bmu_coord[0] - radius, bmu_coord[0] + radius + 1):
            for j in range(bmu_coord[1] - radius, bmu_coord[1] + radius + 1):
                # Check neighbour coordinate in the map not winner coordinates
                if (i, j) in self.map and (i, j) != (bmu_coord[0], bmu_coord[1]):
                    distance = self.lattice_distance((bmu_coord[0], bmu_coord[1]), (i, j))
                    if distance <= radius:
                        neighbors.append(((i, j), distance))
        return neighbors

    def _spread_error(self, x, y):
        """Distribute error to neighbors using Gaussian function"""
        error = self.node_errors[self.map[(x, y)]]
        self.node_errors[self.map[(x, y)]] = error / 2  # make the winner error half
        
        # Distribute half of error to neighbours i.e. error of BMU will ripple outwards to its immediate neighbours
        neighbors = self.get_lattice_neighbors((x, y), 1)
        if len(neighbors) == 0:
            return
            
        total_influence = sum([self.gussian_neighbourhood_function(n[1], 1) for n in neighbors])
        for (i, j), distance in neighbors:
            influence_factor = self.gussian_neighbourhood_function(distance, 1) / total_influence
            self.node_errors[self.map[(i, j)]] += (error / 2) * influence_factor

    def _grow_and_error_distribute(self, x, y, bmu_index):
        """Check for growth and distribute error"""
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        
        if (leftx, lefty) in self.map and (rightx, righty) in self.map and \
           (topx, topy) in self.map and (bottomx, bottomy) in self.map:
            self._spread_error(x, y)
        else:
            self._grow_node(x, y, leftx, lefty, 0)
            self._grow_node(x, y, rightx, righty, 1)
            self._grow_node(x, y, topx, topy, 2)
            self._grow_node(x, y, bottomx, bottomy, 3)
            # Distribute error to all existing neighbors (including newly added ones).
            self._spread_error(x, y)
            # However, the original paper does not distribute the error of the newly added nodes.
            #self.node_errors[bmu_index] = self.groth_threshold/2 

    def _winner_identification_and_weight_adaptation(self, data_index, data, radius, learning_rate):
        """Find BMU and update neighborhood for mixed data"""
        data_point = data[data_index]
        
        # Find BMU using mixed distance
        min_dist = float('inf')
        bmu_index = -1
        for i in range(self.node_count):
            dist = self._compute_mixed_distance(data_point, i)
            if dist < min_dist:
                min_dist = dist
                bmu_index = i
        
        error_val = min_dist
        bmu_x = int(self.node_coordinate[bmu_index][0])
        bmu_y = int(self.node_coordinate[bmu_index][1])
        
        # Update winner weights
        # Numerical features
        num_data = data_point[:self.num_dimensions]
        num_error = num_data - self.node_list_num[bmu_index]
        self.node_list_num[bmu_index] += num_error * learning_rate
        
        # Categorical features - update probabilities
        for feat_idx in self.cat_features:
            category = int(data_point[feat_idx])
            # Increase probability of observed category
            self.node_list_cat[bmu_index][feat_idx] *= (1 - learning_rate)
            self.node_list_cat[bmu_index][feat_idx][category] += learning_rate
            # Normalize
            self.node_list_cat[bmu_index][feat_idx] /= self.node_list_cat[bmu_index][feat_idx].sum()
        
        # Update neighborhood using Gaussian neighborhood function
        # SOM learning rule: wi(t+1) = wi(t) + η(t) × h(t) × (xj - wi(t))
        # where η(t) is learning_rate, h(t) is Gaussian neighborhood function
        mask_size = round(radius)
        neighbors = self.get_lattice_neighbors((bmu_x, bmu_y), mask_size)
        
        # Iterate over the neighbors within the radius
        for (i, j), distance in neighbors:
            neighbor_idx = self.map[(i, j)]
            
            # Gaussian neighborhood function
            h = self.gussian_neighbourhood_function(distance, radius)
            
            # Update numerical weights
            num_diff = self.node_list_num[bmu_index] - self.node_list_num[neighbor_idx]
            self.node_list_num[neighbor_idx] += learning_rate * h * num_diff
            
            # Update categorical probabilities
            for feat_idx in self.cat_features:
                category = int(data_point[feat_idx])
                self.node_list_cat[neighbor_idx][feat_idx] *= (1 - learning_rate * h)
                self.node_list_cat[neighbor_idx][feat_idx][category] += learning_rate * h
                # Normalize
                self.node_list_cat[neighbor_idx][feat_idx] /= \
                    self.node_list_cat[neighbor_idx][feat_idx].sum()
        
        return bmu_index, bmu_x, bmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        """Smoothing phase"""
        for data_index in range(data.shape[0]):
            self._winner_identification_and_weight_adaptation(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        """Growing phase"""
        for data_index in range(data.shape[0]):
            bmu_index, bmu_x, bmu_y, error_val = \
                self._winner_identification_and_weight_adaptation(data_index, data, radius, learning_rate)
            
            # Winner node error update and grow
            self.node_errors[bmu_index] += error_val
            if self.node_errors[bmu_index] > self.groth_threshold:
                self._grow_and_error_distribute(bmu_x, bmu_y, bmu_index)

    def fit(self, data, training_iterations, smooth_iterations):
        """Train the GSOM map"""
        current_learning_rate = self.learning_rate
        
        # Growing iterations
        for i in tqdm(range(training_iterations), desc="Growing"):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self.grow(data, radius_exp, current_learning_rate)
        
        # Smoothing iterations
        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations), desc="Smoothing"):
            radius_exp = self._get_neighbourhood_radius(smooth_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self.smooth(data, radius_exp, current_learning_rate)
        
        # Identify winners
        winners = []
        for data_idx in range(data.shape[0]):
            min_dist = float('inf')
            winner = -1
            for node_idx in range(self.node_count):
                dist = self._compute_mixed_distance(data[data_idx], node_idx)
                if dist < min_dist:
                    min_dist = dist
                    winner = node_idx
            winners.append(winner)
        
        return np.array(winners)

    def predict(self, data, index_col, label_col=None):
        """Predict winning nodes for test dataset"""
        # Prepare dataset
        weight_columns = list(data.columns.values)
        output_columns = [index_col]
        if label_col:
            weight_columns.remove(label_col)
            output_columns.append(label_col)
        weight_columns.remove(index_col)
        
        data_n = data[weight_columns].to_numpy()
        data_out = pd.DataFrame(data[output_columns])
        
        # Identify winners
        winners = []
        for data_idx in range(data_n.shape[0]):
            min_dist = float('inf')
            winner = -1
            for node_idx in range(self.node_count):
                dist = self._compute_mixed_distance(data_n[data_idx], node_idx)
                if dist < min_dist:
                    min_dist = dist
                    winner = node_idx
            winners.append(winner)
        
        data_out["output"] = winners
        
        grp_output = data_out.groupby("output")
        dn = grp_output[index_col].apply(list).reset_index()
        dn = dn.set_index("output")
        if label_col:
            dn[label_col] = grp_output[label_col].apply(list)
        dn = dn.reset_index()
        dn["hit_count"] = dn[index_col].apply(lambda x: len(x))
        dn["x"] = dn["output"].apply(lambda x: self.node_coordinate[x, 0])
        dn["y"] = dn["output"].apply(lambda x: self.node_coordinate[x, 1])
        
        self.node_labels = dn
        self.output = data_out
        
        return self.node_labels


if __name__ == '__main__':
    np.random.seed(1)
    df = pd.read_csv(data_filename)
    print(f"Data shape: {df.shape}")
    
    # Example: First 13 features are numerical, features 13-16 are categorical
    # Modify according to your actual data structure
    num_features = 13
    cat_features_dict = {13: 5, 14: 3, 15: 2, 16: 7}  # {feature_idx: num_categories}
    
    data_training = df.iloc[:, 1:17].to_numpy()
    
    gsom = GSOM_Mixed(
        spred_factor=0.83,
        num_dimensions=num_features,
        cat_dimensions_dict=cat_features_dict,
        max_radius=4
    )
    
    gsom.fit(data_training, 100, 50)
    output = gsom.predict(df, "Name", "label")
    output.to_csv("output_mixed.csv", index=False)
    
    print(f"Training complete. Total nodes: {gsom.node_count}")