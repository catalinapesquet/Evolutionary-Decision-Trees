# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:04:10 2025

@author: Aurora
"""
from graphviz import Digraph
import os

def print_tree(node, depth=0, prefix=""):
    indent = "|   " * depth
    if node.is_leaf:
        print(f"{indent}|--- class: {node.leaf_value}")
    else:
        print(f"{indent}|--- feature_{node.feat_idx} <= {node.threshold:.2f}")
        print_tree(node.left, depth + 1)
        print(f"{indent}|--- feature_{node.feat_idx} >  {node.threshold:.2f}")
        print_tree(node.right, depth + 1)


def export_tree_dot(node, output_name="decision_tree"):

    dot = Digraph(comment='Decision Tree', format='png')
    dot.attr('node', shape='box', style='filled', fontname="helvetica")
    
    def get_color(gain):
        if gain < 0.6:
            return "#A2D5AB"  # green
        elif gain < 1.2:
            return "#FFCC99"  # orange 
        else:
            return "#FF9999"  # red

    def recurse(node):
        node_id = str(id(node))

        if node.is_leaf:
            label = f"class: {node.leaf_value}"
            color = "#B3E5FC"  
        else:
            label = (f"x[{node.feat_idx}] <= {node.threshold:.3f}\n"
         f"gain: {node.gain:.4f}\nsamples: {node.samples}\nvalue: {node.value}")

            color = get_color(node.gain)

        dot.node(node_id, label=label, fillcolor=color)

        if not node.is_leaf:
            # Ajoute les arêtes avec labels
            dot.edge(node_id, str(id(node.left)), label="True")
            dot.edge(node_id, str(id(node.right)), label="False")
            recurse(node.left)
            recurse(node.right)

    recurse(node)
    output_path = dot.render(output_name, cleanup=True)
    os.startfile(output_path)