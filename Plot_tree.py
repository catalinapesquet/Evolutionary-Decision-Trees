# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:04:10 2025
@author: Aurora
"""
from graphviz import Digraph
import numpy as np
import os

def print_tree(node, depth=0):
    indent = "|   " * depth
    if node.is_leaf:
        print(f"{indent}|--- class: {node.leaf_value}")
    else:
        print(f"{indent}|--- feature_{node.feat_idx} <= {node.threshold:.2f}")
        print_tree(node.left, depth + 1)
        print(f"{indent}|--- feature_{node.feat_idx} >  {node.threshold:.2f}")
        print_tree(node.right, depth + 1)

def color_brew(n):
    if n <= 0:
        n = 1  # Sécurité : au moins une couleur pour éviter la division par zéro
    s, v = 0.75, 0.9
    c = s * v
    m = v - c
    color_list = []
    for h in np.arange(25, 385, 360.0 / n).astype(int):
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        rgb = [
            (c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x), (c, x, 0)
        ]
        r, g, b = rgb[int(h_bar)]
        color_list.append([int(255 * (r + m)), int(255 * (g + m)), int(255 * (b + m))])
    return color_list


def get_n_classes(node):
    classes = set()
    def traverse(n):
        if n.class_distribution is not None:
            classes.update(range(len(n.class_distribution)))
        if not n.is_leaf:
            traverse(n.left)
            traverse(n.right)
    traverse(node)
    return len(classes)


def export_tree_dot(node, output_name="decision_tree"):
    dot = Digraph(comment='Decision Tree', format='png')
    dot.attr('node', shape='box', style='filled', fontname="helvetica")

    n_classes = get_n_classes(node)
    raw_colors = color_brew(n_classes)
    class_colors = ['#%02x%02x%02x' % tuple(rgb) for rgb in raw_colors]

    def get_color(n):
        index = np.argmax(n.class_distribution)
        return class_colors[index]

    def recurse(n):
        node_id = str(id(n))
        if n.is_leaf:
            label = f"class: {n.leaf_value}\nsamples: {n.samples}\nvalue: {n.value}"
        else:
            label = (f"x[{n.feat_idx}] <= {n.threshold:.3f}\n"
                     f"gain: {n.gain:.4f}\nsamples: {n.samples}\nvalue: {n.value}")
        color = get_color(n)
        dot.node(node_id, label=label, fillcolor=color)
        if not n.is_leaf:
            dot.edge(node_id, str(id(n.left)), label="True")
            dot.edge(node_id, str(id(n.right)), label="False")
            recurse(n.left)
            recurse(n.right)

    recurse(node)
    output_path = dot.render(output_name, cleanup=True)
    os.startfile(output_path)
