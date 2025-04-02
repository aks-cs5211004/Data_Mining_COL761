import os
import sys
import time
import matplotlib.pyplot as plt
import subprocess


def apriori(s, apriori_path, dataset_path, output_path):
    start_time = time.time()
    try:
        subprocess.run([
        f"{apriori_path}", 
        f"-s{s}", 
        f"{dataset_path}", 
        f"{output_path}/ap{s}"
        ], timeout= 3600,
        check=True
    )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
        os.system(f"rm -rf {output_path}/ap{s}")
        os.system(f"touch -f {output_path}/ap{s}")


    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken

def fpgrowth(s, fp_path, dataset_path, output_path):
    start_time = time.time()
    try:
        subprocess.run([
        f"{fp_path}", 
        f"-s{s}", 
        f"{dataset_path}", 
        f"{output_path}/fp{s}"
        ], timeout= 3600,
        check=True
    )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
        os.system(f"rm -rf {output_path}/fp{s}")
        os.system(f"touch -f {output_path}/fp{s}")

    end_time = time.time()
    time_taken = end_time - start_time
    return time_taken


def plot_graph(s_values, time_taken_aprori, time_taken_fpgrowth, output_path):
    plt.plot(s_values, time_taken_aprori, label="Apriori")
    plt.plot(s_values, time_taken_fpgrowth, label="FPGrowth")
    plt.xlabel("Support (in %)")
    plt.ylabel("Time Taken (in s)")
    plt.title("Time taken vs Support")
    plt.legend()
    plt.savefig(f"{output_path}/plot.png")



if __name__ == "__main__":
    apriori_path = sys.argv[1]
    fp_path = sys.argv[2] 
    dataset_path = sys.argv[3]
    output_path = sys.argv[4]

    s_values =[5,10,25,50,90]
    time_taken_aprori = []
    time_taken_fpgrowth = []
   
    os.system(f"rm -rf {output_path}")
    os.system(f"mkdir -p {output_path}")

    for s in s_values:
        os.system(f"touch {output_path}/ap{s}")
        os.system(f"touch {output_path}/fp{s}")


    for s in s_values:
        time_taken_aprori.append(apriori(s, apriori_path, dataset_path, output_path))
        time_taken_fpgrowth.append(fpgrowth(s, fp_path, dataset_path, output_path))

    plot_graph(s_values, time_taken_aprori, time_taken_fpgrowth,output_path)

    # print(time_taken_aprori)
    # print(time_taken_fpgrowth)

    
