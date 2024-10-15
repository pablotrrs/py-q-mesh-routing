import numpy as np
import random
import json
import os

os.makedirs('logs', exist_ok=True)

# Hyperparameters
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
DECAY_RATE = 0.99
epsilon = INITIAL_EPSILON
ALPHA = 0.5
GAMMA = 0.9
GRID_SIZE = 6

# Define the 6x6 grid topology with 36 intermediary nodes
nodes = ['tx'] + [f'i{n}' for n in range(GRID_SIZE * GRID_SIZE)] + ['rx']

# Define neighbors for the 6x6 irregular grid topology
def generate_neighbors(GRID_SIZE):
    neighbors = {}

    # Add 'tx' and 'rx' neighbors
    neighbors['tx'] = ['i0']
    neighbors['rx'] = [f'i{GRID_SIZE * GRID_SIZE - 1}']
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            current_node = f'i{idx}'
            neighbors[current_node] = []
            
            # Add neighbor to the left (if not on the left edge)
            if j > 0 and idx not in [3, 9, 15, 25, 27, 29]:
                neighbors[current_node].append(f'i{idx - 1}')
            
            # Add neighbor to the right (if not on the right edge)
            if j < GRID_SIZE - 1 and idx not in [2, 8, 14, 24, 26, 28]:
                neighbors[current_node].append(f'i{idx + 1}')
                
            # Add neighbor above (if not on the top row)
            if i > 0 and idx not in [31, 32, 33, 34]:
                neighbors[current_node].append(f'i{idx - GRID_SIZE}')
            
            # Add neighbor below (if not on the bottom row)
            if i < GRID_SIZE - 1 and idx not in [25, 26, 27, 28]:
                neighbors[current_node].append(f'i{idx + GRID_SIZE}')

    neighbors[f'i{GRID_SIZE * GRID_SIZE - 1}'].append('rx')
    
    return neighbors

# Generate neighbors dynamically
neighbors = generate_neighbors(GRID_SIZE)

# Initialize Q-tables, processing times, and node lifetimes
def initialize_q_table():
    q_table = {}
    for node in nodes:
        q_table[node] = {}
        for dest in nodes:
            if dest in neighbors[node]:
                q_table[node][dest] = np.random.rand()
            else:
                q_table[node][dest] = 0
    return q_table

# q_table = {node: {dest: [np.random.rand() if dest in neighbors[node] else 0 for dest in nodes] for dest in nodes} for node in nodes}
q_table = initialize_q_table()

# Save the Q-table to a JSON file in the 'logs' directory
with open('logs/q_table.json', 'w') as f:
    json.dump(q_table, f, indent=4)

processing_time = {node: random.randint(1, 5) for node in nodes}
node_lifetime = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_reconnect_time = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_status = {node: True for node in nodes if node not in ['tx', 'rx']}

# Functions
functions = ["A", "B", "C"]
functions_sequence = ["A", "B", "C"]
nodes_intermediate = [node for node in nodes if node not in ['tx', 'rx']]
node_functions = {}

def assign_functions_to_nodes():
    """Assign functions to intermediary nodes."""
    function_counts = {func: 0 for func in functions}

    for node in nodes_intermediate:
        min_assigned_func = min(function_counts, key=function_counts.get)
        node_functions[node] = min_assigned_func
        function_counts[min_assigned_func] += 1

assign_functions_to_nodes()

def update_node_status():
    """Updates the status of nodes, managing lifetimes and reconnection times."""
    for node in node_status:
        if node_status[node]:
            node_lifetime[node] -= 1
            if node_lifetime[node] <= 0:
                node_status[node] = False
                node_reconnect_time[node] = np.random.exponential(scale=10)
                del node_functions[node]
        else:
            node_reconnect_time[node] -= 1
            if node_reconnect_time[node] <= 0:
                node_status[node] = True
                node_lifetime[node] = np.random.exponential(scale=20)
                function_counts = {func: list(node_functions.values()).count(func) for func in functions}
                min_assigned_func = min(function_counts, key=function_counts.get)
                node_functions[node] = min_assigned_func

def select_next_node(q_values, available_nodes):
    """Selects the next node based on Q-values and exploration/exploitation."""
    available_nodes = [n for n in available_nodes if node_status.get(n, True)]
    if not available_nodes:
        return None

    # Epsilon-greedy strategy: explore with probability epsilon, exploit otherwise
    if random.uniform(0, 1) < epsilon:
        return random.choice(available_nodes)  # Explore: random choice
    else:
        # Exploit: choose the node with the highest Q-value
        max_q_value = max(q_values[nodes.index(n)] for n in available_nodes)
        best_nodes = [n for n in available_nodes if q_values[nodes.index(n)] == max_q_value]
        return random.choice(best_nodes)  # If multiple best nodes, choose randomly

def update_q_value(current_node, next_node, destination, reward):
    """Updates the Q-value for the current state-action pair."""
    with open('logs/q_value_updates.log', 'a') as log_file:
        log_file.write(f"Updating Q-value for transition: {current_node} -> {next_node} -> {destination} with reward {reward}.\n")
        current_q = q_table[current_node][destination]
        log_file.write(f"Current Q-value: {current_q}\n")
        max_next_q = max(q_table[next_node].values())
        log_file.write(f"Next node Q-values: {list(q_table[next_node].values())}\n")
        log_file.write(f"Max Q-value for next node: {max_next_q}\n")
        new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_next_q)
        log_file.write(f"Updated Q-value: {new_q}\n")
        q_table[current_node][destination] = new_q
        log_file.write(f"New Q-value in Q-table: {q_table[current_node][destination]}\n")
        log_file.write(f"---------------------------------------------------------------\n")

def send_packet(tx, rx):
    """Simulates the packet routing process and returns the path, hops, time, and processed functions."""
    global epsilon
    current_node = tx
    total_hops = 0
    total_time = 0
    max_hops = 100

    path = [current_node]
    functions_to_process = functions_sequence.copy()
    processed_functions = []

    while not (current_node == rx and len(functions_to_process) == 0):
        if total_hops >= max_hops:
            print(f"Packet lost after {total_hops} hops.")
            return path, total_hops, total_time, processed_functions

        available_nodes = [n for n in neighbors[current_node] if node_status.get(n, True)]
        next_node = select_next_node(q_table[current_node][rx], available_nodes)

        if next_node is None:
            print(f"Node {current_node} cannot send the packet, no available nodes.")
            return path, total_hops, total_time, processed_functions

        reward = 0

        if functions_to_process:
            expected_function = functions_to_process[0]
            node_function = node_functions.get(next_node, None)

            if node_function == expected_function:
                functions_to_process.pop(0)
                processed_functions.append(node_function)
                reward += 10
            else:
                reward -= 1

        update_q_value(current_node, next_node, rx, reward)

        current_node = next_node
        path.append(current_node)
        total_hops += 1
        total_time += processing_time[current_node]

    epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    return path, total_hops, total_time, processed_functions