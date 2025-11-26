from bschema import create_bschema, Graph, A, bind_prefixes, Namespace, BACNET, S223, RDFS, BRICK, REF, QUDT, QK, UNIT, SKOS, OWL, TAG, SH
import os 
import csv
import matplotlib.pyplot as plt

REC = Namespace("https://w3id.org/rec#")
BSH = Namespace('https://brickschema.org/schema/BrickShape')
VOAG = Namespace('http://voag.linkedmodel.org/schema/voag')
PROV = Namespace('http://www.w3.org/ns/prov')

REMOVE_NAMESPACES = [BACNET, S223, BRICK, REC, REF, QUDT, QK, UNIT, BSH, VOAG, SKOS, OWL, TAG, SH, PROV, OWL]
REMOVE_OBJECTS = [SH.PropertyShape]


print("Loading ontologies...")
BRICK_GRAPH = Graph(store = "Oxigraph")
BRICK_GRAPH.parse("ontologies/Brick.ttl", format = 'ttl')
S223_GRAPH = Graph(store = "Oxigraph")
S223_GRAPH.parse("ontologies/223p.ttl", format = 'ttl')

def remove_triples(g, g2):
    for triple in g2:
        g.remove(triple)

def remove_triples_recursively(g, s):
    if not ((s, None, None) in g):
        return
    for p, o in g.predicate_objects(s):
        g.remove((s, p, o))
        remove_triples_recursively(g, o)

def remove_triples_namespace(g):
    for triple in g:
        for namespace in REMOVE_NAMESPACES:
            if str(namespace) in str(triple[0]):
                remove_triples_recursively(g, triple[0])
        if str(OWL) in str(triple[2]):
            remove_triples_recursively(g, triple[0])

# NOTE: Ref namespace wrong in b59
def rename_b59_nodes(g):
    EX = Namespace('http://data.ashrae.org/standard223/data/lbnl-example-2#')
    REF = Namespace('https://brickschema.org/schema/Brick/')
    for s in g.subjects():
        if (str(EX) in s):
            label = g.value(s, RDFS.label)
            if label == None:
                continue
            if (s, A, REF.APIReference) in g:
                continue
            new_uri = EX[f"{label.replace(' ','_')}{str(s).split('#')[-1]}"]
            for p, o in g.predicate_objects(s):
                g.remove((s,p,o))
                g.add((new_uri, p, o))
            for s2, p2 in g.subject_predicates(s):
                g.remove((s2, p2, s))
                g.add((s2, p2, new_uri))

def undo_brick_inferencing(g):
    for triple in g:
        if triple[1] in [BRICK.Relationship, BRICK.hasTag, REC.isPointOf, REC.hasPoint]:
            g.remove(triple)

def remove_less_specific_classes(g, ontology):
    query = """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        CONSTRUCT {
            ?s rdf:type ?parent .
        }
        WHERE {
            ?s rdf:type ?child .
            ?child rdfs:subClassOf+ ?parent .
            ?s rdf:type ?parent .
        }
    """
    delete_graph = (g + ontology).query(query).graph
    g = g - delete_graph


def write_csv(filename, g_lens, cg_lens):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["graph_length", "bschema_length"])
        for item1, item2 in zip(g_lens, cg_lens):
            writer.writerow([item1, item2])

def process_graphs(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".ttl"):
            if not (file_name in ['b59.ttl']): 
                continue
            file_path = os.path.join(directory_path, file_name)
            print(f"Processing file: {file_name}")
            g = Graph(store = "Oxigraph")
            g.parse(file_path, format = 'ttl')
            print("Removing ontology triples")
            remove_triples_namespace(g)
            if (file_name in ['bldg_brick_anon.ttl', 'bldg11.ttl']):
                remove_less_specific_classes(g, BRICK_GRAPH)
                undo_brick_inferencing(g)
            if file_name in ['b59.ttl']:
                rename_b59_nodes(g)
            g.serialize(f"without-ontology/{file_name}", format="turtle")

if __name__ == "__main__":
    directory_path = "../eval_buildings"
    process_graphs(directory_path)