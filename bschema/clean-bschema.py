#!/usr/bin/env python3
"""
Remove a specific fragment from every URI in a Turtle file
and drop triples whose object is rdfs:Resource.

Usage:
    python clean_ttl.py  INPUT.ttl  [OUTPUT.ttl]

If OUTPUT is omitted the input file is overwritten.
"""

import argparse
import sys
from pathlib import Path
import re

from rdflib import Graph, URIRef, BNode
from rdflib.namespace import RDFS
from bschema.namespaces import bind_prefixes, REF, BRICK, S223
import os 
# ----------------------------------------------------------------------
URN_BSCHEMA = "urn:bschema#http://qudt.org/vocab/quantitykind/"
BRICK_OLD   = "https://brickschema.org/schema/Brick/"
# actually should be ref
BRICK_NEW   = "https://brickschema.org/schema/Brick/ref#"
BRICK_REF   = "https://brickschema.org/schema/Brick/ref#"

BRICK_REGEX = re.compile(
    r"^" + re.escape(BRICK_OLD) + r"(?!ref#)"
)

def clean_uri(uri: URIRef) -> URIRef:
    """
    Apply all URI transformations and return a possibly new URIRef.
    """
    u = str(uri)

    # 1️⃣ Strip the urn:bschema fragment if present at the *start* of the URI.
    if u.startswith(URN_BSCHEMA):
        u = u[len(URN_BSCHEMA) :]

    # 2️⃣ Replace the Brick base with the new version – but only when the URI does
    #    NOT already contain ".../Brick/ref#". The regex above guarantees that.
    if BRICK_REGEX.match(u):
        u = BRICK_NEW + u[len(BRICK_OLD) :]

    # No other changes – return a (potentially) new URIRef.
    return URIRef(u)

def ext_ref_to_bnode(g):
    # If an external reference is only the object of one thing, then we can just make it a blank node
    bnode_mapping = {}
    for s, p, o in g:
        if p in [S223.hasExternalReference, REF.hasExternalReference]:
            if o in bnode_mapping.keys():
                del bnode_mapping[o]
            else:
                bnode = BNode()
                bnode_mapping.update({o:bnode})
    
    for s,p,o in g:
        if o in bnode_mapping.keys():
            g.remove((s, p, o))
            g.add((s, p, bnode_mapping[o]))
        if s in bnode_mapping.keys():
            g.remove((s,p,o))
            g.add((bnode_mapping[s], p, o))

def bind_preset_prefixes(g: Graph):
    bind_prefixes(g)


def process(file_path) -> None:
    g = Graph()
    g.parse(file_path, format="turtle")
    print("Length is :", len(g))

    new_g = Graph()
    for s, p, o in g:
        if o == RDFS.Resource:
            continue

        # clean URIs in subject, predicate, object
        s = clean_uri(s) if isinstance(s, URIRef) else s
        p = clean_uri(p) if isinstance(p, URIRef) else p
        o = clean_uri(o) if isinstance(o, URIRef) else o

        new_g.add((s, p, o))
    bind_preset_prefixes(new_g)
    ext_ref_to_bnode(new_g)
    new_g.serialize(file_path, format="turtle")

# ----------------------------------------------------------------------
def main():
    for dir_name in os.listdir('bschema'):
        for file_name in os.listdir(f'bschema/{dir_name}'):
            if file_name.endswith(".ttl"):
                print(f"Processing file: {dir_name} / {file_name}")
                process(f'bschema/{dir_name}/{file_name}')

if __name__ == "__main__":
    main()