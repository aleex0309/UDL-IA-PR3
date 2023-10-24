import sys
from collections import Counter

from decision_node import DecisionNode

def prune_tree(root: DecisionNode, threshold: float):
    """
    Prune the tree given a certain root and threshold
    """
    if non_leaf(root.tb):
        prune_tree(root.tb, threshold)
    if non_leaf(root.fb):
        prune_tree(root.fb, threshold)
    if both_children_leaf(root):
        if split_quality_smaller_than_threshold(root, threshold):
            _combine_leaves(root)

def non_leaf(node: DecisionNode):
    """
    Check if a node is non-leaf
    """
    return node.results is None

def both_children_leaf(node: DecisionNode):
    """
    Check if both children of a node are leaf nodes
    """
    return node.tb.results is not None and node.fb.results is not None

def split_quality_smaller_than_threshold(node: DecisionNode, threshold: float):
    """
    Check if the quality of a node is smaller than the threshold
    """
    return node.split_quality < threshold

def _combine_leaves(node: DecisionNode):
    """
    Combine the leaves of a node
    """
    node.split_feature = -1
    node.split_value = None
    node.results = node.tb.results + node.fb.results
    node.tb = None
    node.fb = None
    node.split_quality = 0

if __name__ == '__main__':
    sys.exit(-1)

