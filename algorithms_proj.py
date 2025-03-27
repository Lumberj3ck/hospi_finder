#!/usr/bin/env python
# coding: utf-8

# In[2]:


import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Define the city or place you want to retrieve the graph for

def get_graph(place_name):
    # Retrieve the street network graph
    graph = ox.graph_from_place(place_name, network_type='drive')

    # Plot the graph
    fig, ax = ox.plot_graph(graph, figsize=(10, 10), node_size=0, edge_linewidth=0.5)

    # Show the plot
    plt.show()
    return graph


# In[3]:


import heapq

def dijkstra(graph, start):
    # Priority queue to store (distance, node) pairs
    queue = [(0, start)]
    # Dictionary to store the shortest distance to each node
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # Explore neighbors
        for neighbor, edge_data in graph[current_node].items():
            distance = current_distance + edge_data[0].get('length', 1)  # Use edge length as weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return distances

def astar(graph, start, end, heuristic):
    # Priority queue to store (f_score, node) pairs
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Dictionaries to store g_scores and f_scores
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = heuristic(start, end)
    
    came_from = {}
    
    while open_set:
        current_f, current_node = heapq.heappop(open_set)
        
        if current_node == end:
            # Reconstruct path
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path, g_score[end]
        
        for neighbor, edge_data in graph[current_node].items():
            tentative_g = g_score[current_node] + edge_data[0].get('length', 1)
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, float('inf')  # No path found

def haversine_heuristic(node1, node2, graph):
    # Get node coordinates from the graph
    y1, x1 = graph.nodes[node1]['y'], graph.nodes[node1]['x']
    y2, x2 = graph.nodes[node2]['y'], graph.nodes[node2]['x']
    
    # Calculate haversine distance between two points
    return ox.distance.great_circle(y1, x1, y2, x2)


# In[4]:


# Define the origin point (latitude, longitude)
# origin_point = (5, 11.5882)

def find_closest_hospital(graph, hospital_nodes, origin_node, hospitals, algorithm='dijkstra'):
    if algorithm == 'dijkstra':
        # Compute shortest distances from the origin to all nodes
        shortest_distances = dijkstra(graph, origin_node)

        # Find the closest hospital
        closest_hospital = None
        min_distance = float('inf')

        for i, hospital_node in enumerate(hospital_nodes):
            distance = shortest_distances[hospital_node]
            if distance < min_distance:
                min_distance = distance
                closest_hospital = hospitals.iloc[i]
                closest_hospital_node = hospital_node

    elif algorithm == 'astar':
        closest_hospital = None
        min_distance = float('inf')
        closest_hospital_node = None
        
        for i, hospital_node in enumerate(hospital_nodes):
            path, distance = astar(
                graph, 
                origin_node, 
                hospital_node, 
                lambda n1, n2: haversine_heuristic(n1, n2, graph))
            
            if distance < min_distance:
                min_distance = distance
                closest_hospital = hospitals.iloc[i]
                closest_hospital_node = hospital_node

    # Print the closest hospital
    print("Closest Hospital:")
    print(closest_hospital[['name', 'geometry']])
    print(f"Distance: {min_distance:.2f} meters (using {algorithm})")
    return closest_hospital, min_distance, closest_hospital_node


# In[5]:


import matplotlib.pyplot as plt

def show_map_path(graph, closest_hospital, origin_point, origin_node, min_distance, path_nodes=None):
    # Get the centroid or coordinates of the closest hospital
    if closest_hospital.geometry.geom_type == 'Point':
        hospital_coords = (closest_hospital.geometry.y, closest_hospital.geometry.x)
    else:
        hospital_coords = (closest_hospital.geometry.centroid.y, closest_hospital.geometry.centroid.x)

    # Find the nearest node to the closest hospital (if not already provided)
    closest_hospital_node = ox.distance.nearest_nodes(graph, hospital_coords[1], hospital_coords[0])

    # Get the shortest path to the closest hospital
    if path_nodes is None:
        shortest_path = nx.shortest_path(graph, origin_node, closest_hospital_node, weight='length')
    else:
        shortest_path = path_nodes

    # Plot the graph and the shortest path
    fig, ax = ox.plot_graph_route(graph, shortest_path, route_linewidth=6, node_size=0, figsize=(10, 10), show=False, close=False)

    # Plot the origin and closest hospital
    ax.scatter(origin_point[1], origin_point[0], c='blue', s=100, label='Origin', zorder=5)
    ax.scatter(hospital_coords[1], hospital_coords[0], c='red', s=100, label='Closest Hospital', zorder=5)

    # Add legend and title
    plt.legend()
    plt.title(f"Closest Hospital: {closest_hospital['name']} ({min_distance:.2f} meters)", fontsize=15)

    # Show the plot
    plt.show()


# In[6]:


def retrieve_city_graph(city):
    place_name = f"{city}, DE"
    graph = get_graph(place_name)

    # Retrieve all hospitals within the city boundaries
    hospitals = ox.features_from_place(place_name, tags={'amenity': 'hospital'})

    # nodes, edges = ox.graph_to_gdfs(graph)
    # Extract hospital geometries (latitude, longitude)
    # hospital_points = hospitals.geometry.apply(lambda geom: (geom.y, geom.x))
    # Extract centroid or coordinates for each hospital
    hospital_points = hospitals.geometry.apply(
        lambda geom: (geom.centroid.y, geom.centroid.x) if geom.geom_type != 'Point' else (geom.y, geom.x))
    # Find the nearest graph node for each hospital
    hospital_nodes = [ox.distance.nearest_nodes(graph, point[1], point[0]) for point in hospital_points]

    return graph, hospital_nodes, hospitals


# In[8]:


global_cities_cache = {}


# In[7]:


def main(city, point, algorithm='dijkstra'):
    if city in global_cities_cache:
        graph, hospital_nodes, hospitals = global_cities_cache[city]
    else: 
        graph, hospital_nodes, hospitals = retrieve_city_graph(city)
        global_cities_cache[city] = [graph, hospital_nodes, hospitals]

    origin_node = ox.distance.nearest_nodes(graph, point[1], point[0])
    
    if algorithm == 'dijkstra':
        closest_hospital, min_distance, closest_hospital_node = find_closest_hospital(
            graph, hospital_nodes, origin_node, hospitals, algorithm='dijkstra')
        show_map_path(graph, closest_hospital, point, origin_node, min_distance)
    elif algorithm == 'astar':
        closest_hospital, min_distance, closest_hospital_node = find_closest_hospital(
            graph, hospital_nodes, origin_node, hospitals, algorithm='astar')
        # For A*, we need to get the actual path that was found
        path, _ = astar(
            graph, 
            origin_node, 
            closest_hospital_node, 
            lambda n1, n2: haversine_heuristic(n1, n2, graph))
        show_map_path(graph, closest_hospital, point, origin_node, min_distance, path_nodes=path)

points = [ 
    ("Berlin", (52.53628, 13.39388), "dijkstra"), 
    ("Berlin", (52.51546, 13.26861), "astar"), 
    ("Magdeburg", (52.1383, 11.6071), "dijkstra"), 
    ("Magdeburg", (52.09048, 11.61770), "astar"), 
    ("Magdeburg", (52.09043, 11.64911), "dijkstra")
]

for city, point, algorithm in points:
    main(city, point, algorithm)