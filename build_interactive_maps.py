"""
Interactive visualization of PMU graphs using Folium.
"""

import os
import numpy as np
import pandas as pd
import folium
from folium import plugins


def create_interactive_map(coordinates, sensor_ids, A, title='Interactive Graph Map', 
                          results_dir='results/graphs', edge_threshold=0.1, 
                          color_edges_by_weight=True):
    """
    Create an interactive Folium map with sensors and edges.
    
    Parameters:
    -----------
    coordinates : ndarray, shape (n_sensors, 2)
        [latitude, longitude] for each sensor
    sensor_ids : array
        Sensor identifiers
    A : ndarray, shape (n_sensors, n_sensors)
        Adjacency matrix
    title : str
        Title for the map
    results_dir : str
        Directory to save the map
    edge_threshold : float
        Only show edges above this threshold
    color_edges_by_weight : bool
        If True, color edges by weight (red = strong, blue = weak)
    
    Returns:
    --------
    m : folium.Map
        Folium map object
    """
    
    # Calculate map center
    center_lat = coordinates[:, 0].mean()
    center_lon = coordinates[:, 1].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add edges first (so they appear below sensors)
    n_edges_drawn = 0
    
    for i in range(A.shape[0]):
        for j in range(i+1, A.shape[1]):
            weight = A[i, j]
            
            if weight > edge_threshold:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                
                # Color gradient: blue (weak) to red (strong)
                if color_edges_by_weight:
                    # Normalize weight to [0, 1]
                    norm_weight = min(weight / A.max() if A.max() > 0 else 0, 1.0)
                    # Color gradient
                    color = f'rgb({int(255*norm_weight)}, 0, {int(255*(1-norm_weight))})'
                else:
                    color = 'blue'
                
                # Line width based on weight
                line_width = max(0.5, weight * 5)
                
                # Add edge
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color=color,
                    weight=line_width,
                    opacity=0.5,
                    popup=f'Weight: {weight:.3f}'
                ).add_to(m)
                
                n_edges_drawn += 1
    
    # Add sensors as markers
    for i, (lat, lon) in enumerate(coordinates):
        # Degree of this node (how many connections)
        degree = np.sum(A[i] > edge_threshold)
        
        # Marker color based on degree (more connections = redder)
        if degree == 0:
            color = 'gray'
        elif degree < 5:
            color = 'blue'
        elif degree < 10:
            color = 'orange'
        else:
            color = 'red'
        
        popup_text = f"""
        <b>Sensor ID:</b> {sensor_ids[i]}<br>
        <b>Lat:</b> {lat:.4f}<br>
        <b>Lon:</b> {lon:.4f}<br>
        <b>Degree:</b> {degree}
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add title
    title_html = '''
                 <div style="position: fixed; 
                     top: 10px; left: 50px; width: 300px; height: 60px; 
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:16px; padding: 10px">
                 <b>{}</b><br>
                 Edges: {} | Nodes: {}
                 </div>
                 '''.format(title, n_edges_drawn, len(sensor_ids))
    
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def main():
    """Main execution."""
    
    results_dir = 'results/graphs'
    
    # Load sensor metadata
    metadata = pd.read_csv(os.path.join(results_dir, 'sensor_metadata.csv'))
    sensor_ids = metadata['SensorID'].values
    coordinates = metadata[['Latitude', 'Longitude']].values
    
    # Load adjacency matrices
    A_geo = np.load(os.path.join(results_dir, 'A_geo.npy'))
    A_corr = np.load(os.path.join(results_dir, 'A_corr.npy'))
    A_hybrid = np.load(os.path.join(results_dir, 'A_hybrid.npy'))
    
    print("Creating interactive maps...")
    
    # Create interactive maps for each graph
    maps = [
        (A_geo, 'Geographic Graph (k-NN)', 'map_geographic_interactive.html', 0.2),
        (A_corr, 'Correlation Graph', 'map_correlation_interactive.html', 0.3),
        (A_hybrid, 'Hybrid Graph', 'map_hybrid_interactive.html', 0.2)
    ]
    
    for A, title, filename, threshold in maps:
        print(f"  Creating {filename}...")
        m = create_interactive_map(coordinates, sensor_ids, A, title=title,
                                   results_dir=results_dir, edge_threshold=threshold)
        
        output_path = os.path.join(results_dir, filename)
        m.save(output_path)
        print(f"  Saved: {filename}")
    
    print(f"\nInteractive maps saved to: {results_dir}")


if __name__ == '__main__':
    main()
