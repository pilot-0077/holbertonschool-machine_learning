#!/usr/bin/env python3
"""
Module that defines classes for building a decision tree.
"""

import numpy as np


class Node:
    """
    Class that represents a node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize a node.

        Args:
            feature: feature index used for the split
            threshold: threshold value used for the split
            left_child: left child node
            right_child: right child node
            is_root: indicates if the node is the root
            depth: depth of the node in the tree
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Return the maximum depth below the node.
        """
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node.

        Args:
            only_leaves: if True, count only leaves

        Returns:
            The number of nodes below this node
        """
        if only_leaves:
            count = 0
        else:
            count = 1

        count += self.left_child.count_nodes_below(only_leaves=only_leaves)
        count += self.right_child.count_nodes_below(only_leaves=only_leaves)
        return count

    def __str__(self):
        """
        Return the string representation of the node.
        """
        node_type = "root" if self.is_root else "node"
        left_str = str(self.left_child)
        right_str = str(self.right_child)

        return "{} [feature={}, threshold={}]\n" \
               "    +---> {}\n" \
               "    +---> {}".format(node_type, self.feature,
                                     self.threshold,
                                     left_str.replace("\n", "\n    "),
                                     right_str.replace("\n", "\n    "))

    def get_leaves_below(self):
        """
        Return the list of all leaves below this node.
        """
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())


class Leaf(Node):
    """
    Class that represents a leaf in a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a leaf.

        Args:
            value: value stored in the leaf
            depth: depth of the leaf
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this leaf.

        Args:
            only_leaves: if True, count only leaves

        Returns:
            1
        """
        return 1

    def __str__(self):
        """
        Return the string representation of the leaf.
        """
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """
        Return a list containing this leaf.
        """
        return [self]


class Decision_Tree:
    """
    Class that represents a decision tree.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a decision tree.

        Args:
            max_depth: maximum depth allowed
            min_pop: minimum population allowed in a node
            seed: seed for the random generator
            split_criterion: split criterion to use
            root: root node of the tree
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Return the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the tree.

        Args:
            only_leaves: if True, count only leaves

        Returns:
            The number of nodes in the tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Return the string representation of the tree.
        """
        return str(self.root)

    def get_leaves(self):
        """
        Return the list of all leaves in the tree.
        """
        return self.root.get_leaves_below()
