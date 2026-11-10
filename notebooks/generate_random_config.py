import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import random
    import yaml

    def generate_network_yaml(N, output_filename="network.yaml"):
        """
        Generates a random network structure matching the specified YAML layout.
    
        Parameters:
        - N: Total number of nodes in the network
        - output_filename: Target path for the generated YAML file
        """
        if N < 2:
            raise ValueError("N must be at least 2 to establish connections.")

        # 1. Randomly assign types to each node index (ensuring at least one of each)
        types = [random.choice(["mr", "g"]) for _ in range(N)]
        if "mr" not in types: types[0] = "mr"
        if "g" not in types: types[-1] = "g"

        # 2. Generate random directed edges (sources can be any node, targets must be 'g' nodes)
        g_indices = [i for i, t in enumerate(types) if t == "g"]
        edges = []
    
        # Ensure every 'g' node has at least one incoming regulator
        for target in g_indices:
            # Pick a random source node that isn't the target itself
            possible_sources = [i for i in range(N) if i != target]
            num_inputs = random.randint(1, min(3, len(possible_sources))) # 1 to 3 regulators
            sources = random.sample(possible_sources, num_inputs)
        
            for src in sources:
                edges.append([src, target])
            
        # Sort edges for clean structure
        edges.sort()

        # 3. Build node properties based on their types and in-degree counts
        nodes_list = []
        for i in range(N):
            node_type = types[i]
            node_data = {"type": node_type}
        
            if node_type == "mr":
                # 'mr' nodes get a single positive float for basal_rate
                node_data["basal_rate"] = [round(random.uniform(0.05, 0.5), 5)]
        
            elif node_type == "g":
                # Count incoming edges to determine the size of the 'ki' array
                in_degree = sum(1 for edge in edges if edge[1] == i)
            
                node_data["ki"] = [round(random.uniform(0.5, 5.0), 5) for _ in range(in_degree)]
                node_data["prot_half_life"] = [random.randint(1, 10)]
                node_data["prot_transcription_rate"] = [random.randint(5, 20)]
            
            nodes_list.append(node_data)

        full_data = [nodes_list, edges]

        with open(output_filename, "w") as f:
            yaml.dump(full_data, f, default_flow_style=False, sort_keys=True)
        
        print(f"Successfully generated {output_filename} with N={N} nodes.")

    # Example Usage: Generate a network with 5 nodes
    generate_network_yaml(N=100, output_filename="configs/random_network_100.yaml")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
