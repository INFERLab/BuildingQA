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

def process_graphs(directory_path):
    print("Loading ontologies...")
    brick = Graph(store = "Oxigraph")
    brick.parse("ontologies/Brick.ttl", format = 'ttl')
    s223 = Graph(store = "Oxigraph")
    s223.parse("ontologies/223p.ttl", format = 'ttl')
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".ttl"):
            file_path = os.path.join(directory_path, file_name)
            print(f"Processing file: {file_name}")
            g = Graph(store = "Oxigraph")
            g.parse(file_path, format = 'ttl')
            print("Removing ontology triples")
            remove_triples(g, brick)
            remove_triples(g, s223)
            g.serialize(f"without-ontology/{file_name}", format="turtle")

if __name__ == "__main__":
    directory_path = "../eval_buildings"
    process_graphs(directory_path)