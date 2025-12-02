from bschema.utils import inline_graph
from rdflib import Graph

g = Graph()
g.parse("/home/lazlo/Desktop/semantics/BuildingQA/bschema/bschema/threshold-30/bldg_223p_anon.ttl", format = 'ttl')

g2 = inline_graph(g)

g2.serialize('223p-anon-bnodes.ttl', format = 'ttl')