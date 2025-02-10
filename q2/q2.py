import os
import sys
import time
import matplotlib.pyplot as plt
import subprocess
import shutil
import glob


TOTAL_GRAPHS = 64110
ele = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si']


# ---------------------- GRAPH CONVERTER ---------------------------------
def convert_graph(input_file, output_folder):
    output_file = f"{output_folder}/conv_graph.txt"
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        lines = infile.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Start of a new graph
            if line.startswith("#"):
                graph_id = line[1:]  # Extract graph ID
                outfile.write(f"t # {graph_id}\n")
                i += 1
                # Read number of nodes
                num_nodes = int(lines[i].strip())
                i += 1
                # Read node labels and write them in gSpan format
                for node_id in range(num_nodes):
                    node_label = lines[i].strip()
                    id = ele.index(node_label)
                    if not node_label:  # Handle empty or malformed node labels
                        print(f"Warning: Skipping empty node label at line {i + 1}")
                        continue
                    outfile.write(f"v {node_id} {id}\n")
                    i += 1

                # Read number of edges
                num_edges = int(lines[i].strip())
                i += 1

                # Read edges and write them in gSpan format
                for _ in range(num_edges):
                    
                    edge_data = lines[i].strip().split(" ")
                   
                    if len(edge_data) != 3:  # Check for malformed edge data
                        print(f"Warning: Skipping malformed edge line at line {i + 1}: {lines[i].strip()}")
                        i += 1
                        continue
                    
                    source_node = edge_data[0].strip()
                    dest_node = edge_data[1].strip()
                    edge_label = edge_data[2].strip()
                    
                    if not source_node.isdigit() or not dest_node.isdigit():
                        print(f"Warning: Skipping invalid edge indices at line {i + 1}: {lines[i].strip()}")
                        i += 1
                        continue
                    
                    outfile.write(f"e {source_node} {dest_node} {edge_label}\n")
                    i += 1
            else:
                i += 1

    return output_file





# ---------------------- GSPAN ---------------------------------
def gspan(gspan_path, s, dataset_path, output_path):
    #gspan takes in fraction of support , 
    start_time = time.time()
    
    try:
        subprocess.run([gspan_path,  f"-s{s/100}",  f"-f{dataset_path}", "-o"], timeout= 3600, check=True)
        # os.system(f"timeout 3600s {gspan_path} -s{s/100} -f {dataset_path} -o  {output_path}/gspan{s} ")
        if os.path.exists(f"{dataset_path}.fp"):
            shutil.move(f"{dataset_path}.fp", f"{output_path}/gspan{s}")


    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        if os.path.exists(f"{dataset_path}.fp"):
            os.remove(f"{dataset_path}.fp")
        output_file = f"{output_path}/gspan{s}"
        if os.path.exists(output_file):
            os.remove(output_file)
        open(output_file, 'a').close()  # Equivalent to `touch`

    
    # os.system(f"mv {dataset_path}.fp {output_path}/gspan{s}")
    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken

# ---------------------- GASTON ---------------------------------
def gaston(gaston_path,s, dataset_path, output_path):
    #gaston takes in absolute support
    start_time = time.time()
    # os.system(f"timeout 3600s {gaston_path} {(s/100) * TOTAL_GRAPHS} {dataset_path} {output_path}/gaston{s}")
    try:
        subprocess.run([
        gaston_path, 
        f"{(s/100) * TOTAL_GRAPHS}", 
        dataset_path, 
        f"{output_path}/gaston{s}",
        ], timeout= 3600,
        check=True)
        # os.system(f"timeout 3600s {gaston_path} {(s/100) * TOTAL_GRAPHS} {dataset_path} {output_path}/gaston{s}")
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
        output_file = f"{output_path}/gspan{s}"

        # Remove the file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)

        # Create an empty file (equivalent to `touch`)
        open(output_file, 'a').close()



    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken

# ---------------------- FSG ---------------------------------
def fsg(fsg_path,s, dataset_path, output_path):
    #fsg takes in percentage
    start_time = time.time()
    # os.system(f"timeout 3600s {fsg_path} -s{s} {dataset_path} {output_path}/fsg{s}")
    try:
        subprocess.run([
        fsg_path, 
        f"-s{s}", 
        dataset_path, 
        f"{output_path}/fsg{s}",
        ], timeout= 3600,
        check=True)

        path = os.path.dirname(f"{dataset_path}")
        # subprocess.run(f"[ -f {path}/*.fp ] && mv {path}/*.fp {output_path}/fsg{s}")
        output_dir = os.path.join(output_path, f"fsg{s}")

        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Find all .fp files in the directory
        fp_files = glob.glob(os.path.join(path, "*.fp"))

        # Move each file individually
        for file in fp_files:
            shutil.move(file, output_dir)

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
        path = os.path.dirname(f"{dataset_path}")
        for file in glob.glob(os.path.join(path, "*.fp")):
            os.remove(file)

        # Define the output file path
        output_file = os.path.join(output_path, f"fsg{s}")

        # Remove the file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)

        # Create an empty file (equivalent to `touch`)
        open(output_file, 'a').close()
        

    # os.system(f"mv {path}/*.fp {output_path}/fsg{s}")

    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken


def plot_graph(output_path,s_values, time_taken_gspan, time_taken_gaston, time_taken_fsg):
    plt.plot(s_values, time_taken_gspan, label="GSpan")
    plt.plot(s_values, time_taken_gaston, label="Gaston")
    plt.plot(s_values, time_taken_fsg, label="FSG")
    plt.xlabel("Support (in %)")
    plt.ylabel("Time Taken (in s)")
    plt.title("Time taken vs Support")
    plt.legend()
    plt.savefig(f"{output_path}/graph.png")


if __name__ == "__main__":
    gspan_path = sys.argv[1]
    fsg_path = sys.argv[2]
    gaston_path = sys.argv[3]
    dataset_path = sys.argv[4]
    output_path = sys.argv[5]

    s_values = [5,10,25,50,95]
    time_taken_gspan = []
    time_taken_gaston = []
    time_taken_fsg = []

    parent_data_path = os.path.dirname(f"{dataset_path}")
    #convert graphs
    conv_file = convert_graph(dataset_path,parent_data_path)

    #remove output if already present
    os.system(f"rm -rf {output_path}")
    #create the output directory and files a5, a10, a25, a50, a90
    os.system(f"mkdir -p {output_path}")

    for s in s_values:
        os.system(f"touch {output_path}/gspan{s}")
        os.system(f"touch {output_path}/gaston{s}")
        os.system(f"touch {output_path}/fsg{s}")

    for s in s_values:
        time_taken_gspan.append(gspan(gspan_path, s, conv_file, output_path))
        time_taken_gaston.append(gaston(gaston_path,s, conv_file, output_path))
        time_taken_fsg.append(fsg(fsg_path,s, conv_file, output_path))

    plot_graph(output_path,s_values, time_taken_gspan, time_taken_gaston, time_taken_fsg)



    #extra writing 
    os.system(f"touch {output_path}/metadata.txt")
    with open(f"{output_path}/metadata.txt", "w") as outfile:
        outfile.write(f"{time_taken_gspan}\n")
        outfile.write(f"{time_taken_gaston}\n")
        outfile.write(f"{time_taken_fsg}\n")
        