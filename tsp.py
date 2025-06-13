import numpy as np
import math


def read_tsp_file(file_path):
    """
    Read TSP file and return coordinates and distance matrix

    Parameters:
    file_path (str): Path to the .tsp file

    Returns:
    tuple: (coordinates, distance_matrix)
    """
    coordinates = []
    dimension = 0
    edge_weight_type = ""

    # Read the TSP file
    with open(file_path, 'r') as f:
        reading_node_section = False

        for line in f:
            line = line.strip()

            if line == "NODE_COORD_SECTION":
                reading_node_section = True
                continue

            if reading_node_section and line != "EOF":
                # Parse node coordinates
                node_data = line.split()
                if len(node_data) >= 3:  # Ensure we have at least 3 values (index, x, y)
                    coordinates.append([float(node_data[1]), float(node_data[2])])

            elif "DIMENSION" in line:
                dimension = int(line.split(":")[1])

            elif "EDGE_WEIGHT_TYPE" in line:
                edge_weight_type = line.split(":")[1].strip()

    coordinates = np.array(coordinates)

    # Generate distance matrix based on the edge weight type
    if edge_weight_type == "EUC_2D":
        dist_matrix = generate_euc_2d_matrix(coordinates)
    elif edge_weight_type == "GEO":
        dist_matrix = generate_geo_matrix(coordinates)
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")

    return coordinates, dist_matrix


def generate_euc_2d_matrix(coordinates):
    """
    Generate Euclidean distance matrix
    """
    num_cities = len(coordinates)
    dist_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                # Calculate Euclidean distance
                diff = coordinates[i] - coordinates[j]
                distance = round(np.sqrt(np.sum(diff * diff)))
                dist_matrix[i][j] = distance

    return dist_matrix


def generate_geo_matrix(coordinates):
    """
    Generate Geographical distance matrix
    """
    num_cities = len(coordinates)
    dist_matrix = np.zeros((num_cities, num_cities))
    RRR = 6378.388  # Earth's radius in kilometers

    # Convert coordinates from degrees to radians
    lat = coordinates[:, 0] * math.pi / 180.0
    lon = coordinates[:, 1] * math.pi / 180.0

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                # Calculate geographical distance
                q1 = np.cos(lon[i] - lon[j])
                q2 = np.cos(lat[i] - lat[j])
                q3 = np.cos(lat[i] + lat[j])
                distance = int(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
                dist_matrix[i][j] = distance

    return dist_matrix


# Example usage
if __name__ == "__main__":
    list_data = [
        'burma14', # 1
        'ulysses16', # 2
        'gr17', # 3
        'gr21', # 4
        'ulysses22', # 5
        'gr24', # 6
        'fri26', # 7
        'dantzig42', # 8
        'att48', # 9
        'gr48', # 10,
        'hk48', # 11
        'rand50', # 12
        'eil51', # 13
        'berlin52', # 14
        'st70' # 15
        'eil76', # 16
        'pr76', # 17
        'gr96', # 18
        'rat99', # 19
        'kroA100', # 20
        'kroB100', # 21
        'kroC100', # 22
        'kroD100', # 23
        'kroE100', # 24
        'eil101', # 25
        'lin105', # 26
        'pr107', # 27
        'pr124', # 28
        'ch130', # 29
        'pr136', # 30
        'pr144', # 31
        'ch150', # 32
        'kroA150', # 33
        'kroB150', # 34
        'pr152', # 35
        'kroA200', # 36
        'kroB200', # 37
        'gr202', # 38
        'tsp225', # 39
        'pr226', # 40
        'pr264', # 41
        'lin318', # 42
        'rd400', # 43
        'pr439', # 44
        'rat575', # 45
    ]

    for i in range(len(list_data)):
        city = list_data[i]
        file_path = "./dataset/tsplib-master/" + city + ".tsp"
        try:
            coordinates, distance_matrix = read_tsp_file(file_path)
            index = i + 1
            np.save('./dataset/distance_matrix/' + str(index) + '_' + city + '.npy', distance_matrix)
            print("Distance Matrix Shape:", distance_matrix.shape)
            print("\nFirst 5x5 elements of distance matrix:")
            print(distance_matrix[:5, :5])
        except Exception as e:
            print(f"Error: {e}")