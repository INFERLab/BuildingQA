from bschema import create_bschema, Graph, A, bind_prefixes
import os 
import csv
import matplotlib.pyplot as plt
from time import time 

def remove_triples(g, g2):
    for triple in g2:
        g.remove(triple)


def write_csv(filename, g_lens, cg_lens):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["graph_length", "bschema_length"])
        for item1, item2 in zip(g_lens, cg_lens):
            writer.writerow([item1, item2])

def get_graphs(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".ttl"):
            file_path = os.path.join(directory_path, file_name)
            print(f"Processing file: {file_name}")
            g = Graph(store = "Oxigraph")
            g.parse(file_path, format = 'ttl')
            yield file_name, g


if __name__ == "__main__":
    directory_path = "without-ontology"
    gs = []
    cgs = []
    g_lens = []
    cg_lens = []
    file_names = []
    # NOTE: code is not optimized at all, run time could easily be 20 times faster... 
    runtimes = []
    iteration_lst = []
    thresholds = [0, 0.3, 0.5, 0.7, None]
    for k, threshold in enumerate(thresholds):
        for file_name, g in get_graphs(directory_path):
            g_lens.append(len(g))
            start_time = time() 
            cg, mg, i = create_bschema(g, iterations=20, similarity_threshold=threshold, use_jaccard = False, use_original_names = False)
            print('Confirming iterations: ', i)
            end_time = time()
            if threshold != None:
                threshold_path = f'threshold-{int(threshold*100)}/'
            else:
                threshold_path = 'full/'
            bschema_file_name = 'bschema/'+ threshold_path + file_name
            os.makedirs(os.path.dirname(bschema_file_name), exist_ok=True) 
            bind_prefixes(cg)
            cg.serialize(bschema_file_name, format="turtle")
            print("compressed to ", len(cg)/len(g)*100, "% of its original size")
            cg_lens.append(len(cg))
            iteration_lst.append(i)
            runtimes.append(end_time - start_time)
            file_names.append(file_name)
        
        csv_name = 'bschema/'+ threshold_path + 'stats.csv'
        with open(csv_name, 'w', newline='') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)

            # Optionally, write a header row
            writer.writerow(["file_name","threshold","graph_length", "bschema_length", "iterations", "runtime"])
            for i0, i1, i2, i3, i4 in zip(file_names, g_lens, cg_lens, iteration_lst, runtimes):
                writer.writerow([i0, threshold, i1, i2, i3, i4])
            write_csv('bschema/'+ threshold_path + 'stats.csv', g_lens, cg_lens)
        # just plotting the ones added in last iteration
        plt.plot(g_lens[-6:], cg_lens[-6:], 'o')
        plt.xlabel("Original Graph Size")
        plt.ylabel("Compressed Graph Size")
        plt.title(f"Building Compression By Graph Size (Threshold: {threshold})")
        plt.savefig('bschema/' + threshold_path + "graph_sizes.png")