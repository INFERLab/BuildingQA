from bschema import create_bschema, Graph, A, bind_prefixes
import os 
import csv
import matplotlib.pyplot as plt

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
    print("Loading ontologies...")
    brick = Graph(store = "Oxigraph")
    brick.parse("ontologies/Brick.ttl", format = 'ttl')
    s223 = Graph(store = "Oxigraph")
    s223.parse("ontologies/223p.ttl", format = 'ttl')
    for file_name in os.listdir(directory_path):
        if file_name != 'TUC_building.ttl':
            continue
        if file_name.endswith(".ttl"):
            file_path = os.path.join(directory_path, file_name)
            print(f"Processing file: {file_name}")
            g = Graph(store = "Oxigraph")
            g.parse(file_path, format = 'ttl')
            print("Removing ontology triples")
            remove_triples(g, brick)
            remove_triples(g, s223)
            g.serialize(f"without-ontology/{file_name}", format="turtle")
            
        yield file_name, g


if __name__ == "__main__":
    directory_path = "../eval_buildings"
    gs = []
    cgs = []
    g_lens = []
    cg_lens = []
    file_names = []
    threshold = None
    for threshold in [0, 0.3, 0.5, 0.7, None]:
        for file_name, g in get_graphs(directory_path):
            g_lens.append(len(g))
            cg, mg = create_bschema(g, iterations=20, similarity_threshold=threshold)
            if threshold != None:
                bschema_file_name = 'bschema/'+ f'threshold-{int(threshold*100)}-' + file_name
            else:
                bschema_file_name = 'bschema/' + file_name
            bind_prefixes(cg)
            cg.serialize(bschema_file_name, format="turtle")
            print("compressed to ", len(cg)/len(g)*100, "% of its original size")
            cg_lens.append(len(cg))
        write_csv('bschema/'+ f'theshold-{int(threshold*100)}-lengths.csv', g_lens, cg_lens)
        break
    plt.plot(g_lens, cg_lens, 'o')
    plt.xlabel("Original Graph Size")
    plt.ylabel("Compressed Graph Size")
    plt.title(f"Building Compression By Graph Size (Threshold: {threshold})")
    plt.savefig("graph_sizes.png")
    plt.clf()
    plt.plot(g_lens, [(cg_len/g_len) for cg_len, g_len in zip(cg_lens, g_lens)], 'o')
    plt.xlabel("Original Graph Size")
    plt.ylabel("Compression Ratio")
    plt.title(f"Building Compression Ratio By Graph Size (Threshold: {threshold})")
    plt.savefig("compression_ratios.png")