#!/usr/bin/env python3
import sys
import collections
from math import log2
import random
from typing import List, Tuple

import decision_node
import evaluation
import pruning
from Stack import Stack
from decision_node import DecisionNode

# Used for typing
Data = List[List]


def read(file_name: str, separator: str = ",") -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    header = None
    data = []
    with open(file_name, "r") as fh:
        for line in fh:
            values = line.strip().split(separator)
            if header is None:
                header = values
                continue
            data.append([_parse_value(v) for v in values])
    return header, data

def _parse_value(v):
    try:
        if float(v) == int(v):
            return int(v)
        else:
            return float(v)
    except ValueError:
        return v

def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result)
    """
    return collections.Counter([row[-1] for row in part])

    # Alternative using the counter directly

    # results = collections.Counter()
    # for row in part:
    #    c = row[-1]
    #    results[c] += 1
    # return dict(results)
    # results = {}

def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0

    results = unique_counts(part)
    imp = 1
    for count in results.values():
        probability = count / total # Probability of each class
        imp -= probability ** 2 # Subtract the probability from 1
    return imp # Return the Gini index


def entropy(rows: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    total = len(rows)
    results = unique_counts(rows)
    return -sum(
        (v / total) * log2(v / total) for v in results.values()
    )


def _split_numeric(prototype: List, column: int, value):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value: str):
    return prototype[column] == value
    # raise NotImplementedError


def divideset(part: Data, column: int, value) -> Tuple[Data, Data]: # data, column to check, value of column
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical
    # Split "part" according "split_function"
    set1, set2 = [], []
    for row in part: # For each row in the dataset
        if split_function(row, column, value): # If it matches the criteria
            set1.append(row) # Add it to the first set
        else:
            set2.append(row) # Add it to the second set
    return (set1, set2) # Return both sets


def buildtree(part: Data, scoref=entropy, beta=0): #Recursive version
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0:
        return decision_node.DecisionNode()

    current_score = scoref(part)
    if current_score == 0:
        return decision_node.DecisionNode(results=unique_counts(part), split_quality=0) # Pure node

    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(part[0]) - 1 # -1 because the last column is the label
    for col in range(0, column_count): # Search the best parameters to use
        column_values = {}
        for row in part:
            column_values[row[col]] = 1
        for value in column_values.keys():
            (set1, set2) = divideset(part, col, value)
            p = float(len(set1)) / len(part)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > beta:
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=buildtree(best_sets[0]), fb=buildtree(best_sets[1]), split_quality=best_gain)
    else:
        return DecisionNode(results=unique_counts(part), split_quality=best_gain)


def iterative_buildtree(part: Data, scoref=entropy, beta=0):
    """
    t10: Define the iterative version of the function buildtree
    """

    if len(part) == 0:
        return decision_node.DecisionNode(results=unique_counts(part), split_quality=0) # Pure node

    stack = Stack()
    node_stack = Stack()
    stack.push((0, part, None, 0))
    while not stack.is_empty():
        level, data, criteria, split_quality = stack.pop()
        if level == 0:
            current_score = scoref(data)
            if current_score == 0:
               node_stack.push(DecisionNode(results=unique_counts(data), split_quality=0)) # Pure node
            else:
                best_gain = 0.0
                best_criteria = None
                best_sets = None
                column_count = len(data[0]) - 1
                for col in range(0, column_count): #Search for the best parameters
                    column_values = {}
                    for row in data:
                        column_values[row[col]] = 1
                    for value in column_values.keys():
                        (set1, set2) = divideset(data, col, value)
                        p = float(len(set1)) / len(data)
                        gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
                        if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                            best_gain = gain
                            best_criteria = (col, value)
                            best_sets = (set1, set2)
                if best_gain > beta:
                    stack.push((1, data, best_criteria, best_gain))
                    stack.push((0, best_sets[0], best_criteria, best_gain))
                    stack.push((0, best_sets[1], best_criteria, best_gain))
                else:
                    node_stack.push(DecisionNode(results=unique_counts(data)))
        elif level == 1:
            true_branch = node_stack.pop()
            false_branch = node_stack.pop()
            node_stack.push(DecisionNode(col=criteria[0], value=criteria[1], tb=true_branch, fb=false_branch, split_quality=split_quality))
            if len(data) == len(part):
                return node_stack.pop() # Return the root node

def classify(tree, row):
    if tree.results is not None:
        maximum = max(tree.results.values())
        labels = [k for k, v in tree.results.items() if v == maximum]
        return random.choice(labels)
    if isinstance(tree.value, (int, float)):
        if _split_numeric(row, tree.col, tree.value):
            return classify(tree.tb, row)
        else:
            return classify(tree.fb, row)
    else:
        if _split_categorical(row, tree.col, tree.value):
            return classify(tree.tb, row)
        else:
            return classify(tree.fb, row)



def print_tree(tree, headers=None, indent=""):
    """
    t11: Include the following function
    """
    # Is this a leaf node?
    if tree.results is not None:
        print(tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")


def print_data(headers, data):
    colsize = 15
    print('-' * ((colsize + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(colsize), end="|")
    print("")
    print('-' * ((colsize + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(colsize), end="|")
            else:
                print(value.ljust(colsize), end="|")
        print("")
    print('-' * ((colsize + 1) * len(headers) + 1))

def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"

    headers, data = read(filename)

    print_trees(data, headers)
    predict_data(data)
    print_prunning(data, headers)
    testing(data)
    optimal_threshold(data)

def print_trees(data, headers):
    print("----- TREES -----")
    tree = buildtree(data)
    print("   - RECURSIVE -   ")
    print_tree(tree, headers)
    print("\n\n")
    print("   - ITERATIVE -   ")
    it_tree = iterative_buildtree(data)
    print_tree(it_tree, headers)


def predict_data(data):
    print("----- PREDICTIONS -----")
    train, test = evaluation.train_test_split(data, 0.2)
    tree = buildtree(train)
    for row in test:
        prediction = classify(tree, row[:-1])
        print("Prediction for ", row, "is: ", prediction)


def print_prunning(data, headers):
    print("----- PRUNNING -----")
    tree = buildtree(data)
    print("  - Not pruned -   ")
    print_tree(tree, headers)
    pruning.prune_tree(tree, 0.85)
    print("\n\n")
    print("   - Pruned -    ")
    print_tree(tree, headers)


def testing(data):
    print("----- TESTING -----")
    train, test = evaluation.train_test_split(data, 0.2)
    tree = buildtree(train)
    print("Data split between train and test with 0.2 test size")
    train_accuracy = evaluation.get_accuracy(tree, train)
    print("Accuracy with training data: " + "{:.2f}".format(train_accuracy * 100) + " %")
    test_accuracy = evaluation.get_accuracy(tree, test)
    print("Accuracy with testing data: " + "{:.2f}".format(test_accuracy * 100) + " %")


def optimal_threshold(data):
    print("----- OPTIMAL THRESHOLD -----")
    best_threshold = (None, -1)
    train, test = evaluation.train_test_split(data, 0.2)

    iterations = 15

    segment_division = 1 / iterations
    for i in range(iterations + 1):
        threshold = segment_division * i # 0.0, 0.1, 0.2, ..., 1.0
        result = evaluation.cross_validation(dataset=train, k=5, threshold=threshold)
        if result > best_threshold[1]:
            best_threshold = (threshold, result)

    print(
        "Best threshold is: " + str(best_threshold[0]) +
        ", with an accuracy of " + "{:.2f}".format(best_threshold[1]))

    best_threshold_model = buildtree(train)

    not_prunned_accuracy = evaluation.get_accuracy(best_threshold_model, test)
    print(
        "Accuracy with test dataset on tree "
        "trained with training dataset and not prunned: "
        + "{:.2f}".format(not_prunned_accuracy * 100) + " %")

    pruning.prune_tree(best_threshold_model, best_threshold[0])
    best_threshold_accuracy = evaluation.get_accuracy(best_threshold_model, test)
    print(
        "Accuracy found with test dataset, "
        "on tree trained with training dataset and pruned with the best threshold "
        "found is: "
        + "{:.2f}".format(best_threshold_accuracy * 100) + " %")


if __name__ == "__main__":
    main()