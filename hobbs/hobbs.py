# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

"""
Implementation of Hobbs' algorithm for pronoun resolution.
Chris Ward, 2014
"""

import queue
import sys

import nltk
from nltk import Tree
from nltk.corpus import names

# Labels for nominal heads
nominal_labels = ["NN", "NNS", "NNP", "NNPS", "PRP"]


def get_pos(tree, node):
    """ Given a tree and a node, return the tree position
    of the node.
    """
    for pos in tree.treepositions():
        if tree[pos] == node:
            return pos


def get_dom_np(sents, pos):
    """ Finds the position of the NP that immediately dominates
    the pronoun.
    Args:
        sents: list of trees (or tree) to search
        pos: the tree position of the pronoun to be resolved
    Returns:
        tree: the tree containing the pronoun
        dom_pos: the position of the NP immediately dominating
            the pronoun
    """
    # start with the last tree in sents
    tree = sents[-1]
    # get the NP's position by removing the last element from
    # the pronoun's
    dom_pos = pos[:-1]
    return tree, dom_pos


def walk_to_np_or_s(tree, pos):
    """ Takes the tree being searched and the position from which
    the walk up is started. Returns the position of the first NP
    or S encountered and the path taken to get there from the
    dominating NP. The path consists of a list of tree positions.
    Args:
        tree: the tree being searched
        pos: the position from which the walk is started
    Returns:
        path: the path taken to get the an NP or S node
        pos: the position of the first NP or S node encountered
    """
    path = [pos]
    still_looking = True
    while still_looking:
        # climb one level up the tree by removing the last element
        # from the current tree position
        pos = pos[:-1]
        path.append(pos)
        # if an NP or S node is encountered, return the path and pos
        if "NP" in tree[pos].label() or tree[pos].label() == "S":
            still_looking = False
    return path, pos


def bft(tree):
    """ Perform a breadth-first traversal of a tree.
    Return the nodes in a list in level-order.
    Args:
        tree: a tree node
    Returns:
        lst: a list of tree nodes in left-to-right level-order
    """
    lst = []
    q = queue.Queue()
    q.put(tree)
    while not q.empty():
        node = q.get()
        lst.append(node)
        for child in node:
            if isinstance(child, nltk.Tree):
                q.put(child)
    return lst


def count_np_nodes(tree):
    """ Function from class to count NP nodes.
    """
    np_count = 0
    if not isinstance(tree, nltk.Tree):
        return 0
    elif "NP" in tree.label() and tree.label() not in nominal_labels:
        return 1 + sum(count_np_nodes(c) for c in tree)
    else:
        return sum(count_np_nodes(c) for c in tree)


def check_for_intervening_np(tree, pos, proposal, pro):
    """ Check if subtree rooted at pos contains at least
    three NPs, one of which is:
        (i)   not the proposal,
        (ii)  not the pronoun, and
        (iii) greater than the proposal
    Args:
        tree: the tree being searched
        pos: the position of the root subtree being searched
        proposal: the position of the proposed NP antecedent
        pro: the pronoun being resolved (string)
    Returns:
        True if there is an NP between the proposal and the  pronoun
        False otherwise
    """
    bf = bft(tree[pos])
    bf_pos = [get_pos(tree, node) for node in bf]

    if count_np_nodes(tree[pos]) >= 3:
        for node_pos in bf_pos:
            if "NP" in tree[node_pos].label() \
                and tree[node_pos].label() not in nominal_labels:
                if node_pos != proposal and node_pos != get_pos(tree, pro):
                    if node_pos < proposal:
                        return True
    return False


def traverse_left(tree, pos, path, pro, check=1):
    """ Traverse all branches below pos to the left of path in a
    left-to-right, breadth-first fashion. Returns the first potential
    antecedent found.

    If check is set to 1, propose as an antecedent any NP node
    that is encountered which has an NP or S node between it and pos.
    If check is set to 0, propose any NP node encountered as the antecedent.
    Args:
        tree: the tree being searched
        pos: the position of the root of the subtree being searched
        path: the path taked to get to pos
        pro: the pronoun being resolved (string)
        check: whether or not there must be an intervening NP
    Returns:
        tree: the tree containing the antecedent
        p: the position of the proposed antecedent
    """
    # get the results of breadth first search of the subtree
    # iterate over them
    breadth_first = bft(tree[pos])

    # convert the treepositions of the subtree rooted at pos
    # to their equivalents in the whole tree
    bf_pos = [get_pos(tree, node) for node in breadth_first]

    if check == 1:
        for p in bf_pos:
            if p < path[0] and p not in path:
                if "NP" in tree[p].label() and match(tree, p, pro):
                    if check_for_intervening_np(tree, pos, p, pro) == True:
                        return tree, p

    elif check == 0:
        for p in bf_pos:
            if p < path[0] and p not in path:
                if "NP" in tree[p].label() and match(tree, p, pro):
                    return tree, p

    return None, None


def traverse_right(tree, pos, path, pro):
    """ Traverse all the branches of pos to the right of path p in a
    left-to-right, breadth-first manner, but do not go below any NP
    or S node encountered. Propose any NP node encountered as the
    antecedent. Returns the first potential antecedent.
    Args:
        tree: the tree being searched
        pos: the position of the root of the subtree being searched
        path: the path taken to get to pos
        pro: the pronoun being resolved (string)
    Returns:
        tree: the tree containing the antecedent
        p: the position of the antecedent
    """
    breadth_first = bft(tree[pos])
    bf_pos = [get_pos(tree, node) for node in breadth_first]

    for p in bf_pos:
        if p > path[0] and p not in path:
            if "NP" in tree[p].label() or tree[p].label() == "S":
                if "NP" in tree[p].label() and tree[p].label() not in nominal_labels:
                    if match(tree, p, pro):
                        return tree, p
                return None, None


def traverse_tree(tree, pro):
    """ Traverse a tree in a left-to-right, breadth-first manner,
    proposing any NP encountered as an antecedent. Returns the
    tree and the position of the first possible antecedent.
    Args:
        tree: the tree being searched
        pro: the pronoun being resolved (string)
    """
    # Initialize a queue and enqueue the root of the tree
    q = queue.Queue()
    q.put(tree)
    while not q.empty():
        node = q.get()
        # if the node is an NP, return it as a potential antecedent
        if "NP" in node.label() and match(tree, get_pos(tree, node), pro):
            return tree, get_pos(tree, node)
        for child in node:
            if isinstance(child, nltk.Tree):
                q.put(child)
    # if no antecedent is found, return None
    return None, None


def match(tree, pos, pro):
    """ Takes a proposed antecedent and checks whether it matches
    the pronoun in number and gender

    Args:
        tree: the tree in which a potential antecedent has been found
        pos: the position of the potential antecedent
        pro: the pronoun being resolved (string)
    Returns:
        True if the antecedent and pronoun match
        False otherwise
    """
    if number_match(tree, pos, pro) and gender_match(tree, pos, pro):
        return True
    return False


def number_match(tree, pos, pro):
    """ Takes a proposed antecedent and pronoun and checks whether
    they match in number.
    """
    m = {"NN": "singular",
         "NNP": "singular",
         "he": "singular",
         "she": "singular",
         "him": "singular",
         "her": "singular",
         "it": "singular",
         "himself": "singular",
         "herself": "singular",
         "itself": "singular",
         "NNS": "plural",
         "NNPS": "plural",
         "they": "plural",
         "them": "plural",
         "themselves": "plural",
         "PRP": None}

    # if the label of the nominal dominated by the proposed NP and
    # the pronoun both map to the same number feature, they match
    for c in tree[pos]:
        if isinstance(c, nltk.Tree) and c.label() in nominal_labels:
            if m[c.label()] == m[pro]:
                return True
    return False


def gender_match(tree, pos, pro):
    """ Takes a proposed antecedent and pronoun and checks whether
    they match in gender. Only checks for mismatches between singular
    proper name antecedents and singular pronouns.
    """
    male_names = (name.lower() for name in names.words('male.txt'))
    female_names = (name.lower() for name in names.words('female.txt'))
    male_pronouns = ["he", "him", "himself"]
    female_pronouns = ["she", "her", "herself"]
    neuter_pronouns = ["it", "itself"]

    for c in tree[pos]:
        if isinstance(c, nltk.Tree) and c.label() in nominal_labels:
            # If the proposed antecedent is a recognized male name,
            # but the pronoun being resolved is either female or
            # neuter, they don't match
            if c.leaves()[0].lower() in male_names:
                if pro in female_pronouns:
                    return False
                elif pro in neuter_pronouns:
                    return False
            # If the proposed antecedent is a recognized female name,
            # but the pronoun being resolved is either male or
            # neuter, they don't match
            elif c.leaves()[0].lower() in female_names:
                if pro in male_pronouns:
                    return False
                elif pro in neuter_pronouns:
                    return False
            # If the proposed antecedent is a numeral, but the
            # pronoun being resolved is not neuter, they don't match
            elif c.leaves()[0].isdigit():
                if pro in male_pronouns:
                    return False
                elif pro in female_pronouns:
                    return False

    return True


def hobbs(sents, pos):
    """ The implementation of Hobbs' algorithm.
    Args:
        sents: list of sentences to be searched
        pos: the position of the pronoun to be resolved
    Returns:
        proposal: a tuple containing the tree and position of the
            proposed antecedent
    """
    # The index of the most recent sentence in sents
    sentence_id = len(sents) - 1
    # The number of sentences to be searched
    num_sents = len(sents)

    # Step 1: begin at the NP node immediately dominating the pronoun
    tree, pos = get_dom_np(sents, pos)

    # String representation of the pronoun to be resolved
    pro = tree[pos].leaves()[0].lower()

    # Step 2: Go up the tree to the first NP or S node encountered
    path, pos = walk_to_np_or_s(tree, pos)

    # Step 3: Traverse all branches below pos to the left of path
    # left-to-right, breadth-first. Propose as an antecedent any NP
    # node that is encountered which has an NP or S node between it and pos
    proposal = traverse_left(tree, pos, path, pro)

    while proposal == (None, None):

        # Step 4: If pos is the highest S node in the sentence,
        # traverse the surface parses of previous sentences in order
        # of recency, the most recent first; each tree is traversed in
        # a left-to-right, breadth-first manner, and when an NP node is
        # encountered, it is proposed as an antecedent
        if pos == ():
            # go to the previous sentence
            sentence_id -= 1
            # if there are no more sentences, no antecedent found
            if sentence_id < 0:
                return None
            # search new sentence
            proposal = traverse_tree(sents[sentence_id], pro)
            if proposal != (None, None):
                return proposal

        # Step 5: If pos is not the highest S in the sentence, from pos,
        # go up the tree to the first NP or S node encountered.
        path, pos = walk_to_np_or_s(tree, pos)

        # Step 6: If pos is an NP node and if the path to pos did not pass
        # through the nominal node that pos immediately dominates, propose pos
        # as the antecedent.
        if "NP" in tree[pos].label() and tree[pos].label() not in nominal_labels:
            for c in tree[pos]:
                if isinstance(c, nltk.Tree) and c.label() in nominal_labels:
                    if get_pos(tree, c) not in path and match(tree, pos, pro):
                        proposal = (tree, pos)
                        if proposal != (None, None):
                            return proposal

        # Step 7: Traverse all branches below pos to the left of path,
        # in a left-to-right, breadth-first manner. Propose any NP node
        # encountered as the antecedent.
        proposal = traverse_left(tree, pos, path, pro, check=0)
        if proposal != (None, None):
            return proposal

        # Step 8: If pos is an S node, traverse all the branches of pos
        # to the right of path in a left-to-right, breadth-forst manner, but
        # do not go below any NP or S node encountered. Propose any NP node
        # encountered as the antecedent.
        if tree[pos].label() == "S":
            proposal = traverse_right(tree, pos, path, pro)
            if proposal != (None, None):
                return proposal

    return proposal


def resolve_reflexive(sents, pos):
    """ Resolves reflexive pronouns by going to the first S
    node above the NP dominating the pronoun and searching for
    a matching antecedent. If none is found in the lowest S
    containing the anaphor, then the sentence probably isn't
    grammatical or the reflexive is being used as an intensifier.
    """
    tree, pos = get_dom_np(sents, pos)

    pro = tree[pos].leaves()[0].lower()

    # local binding domain of a reflexive is the lowest clause
    # containing the reflexive and a binding NP
    path, pos = walk_to_s(tree, pos)

    proposal = traverse_tree(tree, pro)

    return proposal


def walk_to_s(tree, pos):
    """ Takes the tree being searched and the position from which
    the walk up is started. Returns the position of the first S
    encountered and the path taken to get there from the
    dominating NP. The path consists of a list of tree positions.
    Args:
        tree: the tree being searched
        pos: the position from which the walk is started
    Returns:
        path: the path taken to get the an S node
        pos: the position of the first S node encountered
    """
    path = [pos]
    still_looking = True
    while still_looking:
        # climb one level up the tree by removing the last element
        # from the current tree position
        pos = pos[:-1]
        path.append(pos)
        # if an S node is encountered, return the path and pos
        if tree[pos].label() == "S":
            still_looking = False
    return path, pos


def demo():
    tree1 = Tree.fromstring('(S (NP (NNP John) ) (VP (VBD said) (SBAR (-NONE- 0) \
        (S (NP (PRP he) ) (VP (VBD likes) (NP (NNS dogs) ) ) ) ) ) )')
    tree2 = Tree.fromstring('(S (NP (NNP John) ) (VP (VBD said) (SBAR (-NONE- 0) \
        (S (NP (NNP Mary) ) (VP (VBD likes) (NP (PRP him) ) ) ) ) ) )')
    tree3 = Tree.fromstring('(S (NP (NNP John)) (VP (VBD saw) (NP (DT a) \
        (JJ flashy) (NN hat)) (PP (IN at) (NP (DT the) (NN store)))))')
    tree4 = Tree.fromstring('(S (NP (PRP He)) (VP (VBD showed) (NP (PRP it)) \
        (PP (IN to) (NP (NNP Terrence)))))')
    tree5 = Tree.fromstring("(S(NP-SBJ (NNP Judge) (NNP Curry))\
        (VP(VP(VBD ordered)(NP-1 (DT the) (NNS refunds))\
        (S(NP-SBJ (-NONE- *-1))(VP (TO to) (VP (VB begin)\
        (NP-TMP (NNP Feb.) (CD 1))))))(CC and)\
        (VP(VBD said)(SBAR(IN that)(S(NP-SBJ (PRP he))(VP(MD would)\
        (RB n't)(VP(VB entertain)(NP(NP (DT any) (NNS appeals))(CC or)\
        (NP(NP(JJ other)(NNS attempts)(S(NP-SBJ (-NONE- *))(VP(TO to)\
        (VP (VB block) (NP (PRP$ his) (NN order))))))(PP (IN by)\
        (NP (NNP Commonwealth) (NNP Edison)))))))))))(. .))")
    tree6 = Tree.fromstring('(S (NP (NNP John) ) (VP (VBD said) (SBAR (-NONE- 0) \
        (S (NP (NNP Mary) ) (VP (VBD likes) (NP (PRP herself) ) ) ) ) ) )')

    print("Sentence 1:")
    print(tree1)
    tree, pos = hobbs([tree1], (1, 1, 1, 0, 0))
    print("Proposed antecedent for 'he':", tree[pos], '\n')

    print(tree2)
    tree, pos = hobbs([tree2], (1, 1, 1, 1, 1, 0))
    print("Proposed antecedent for 'him':", tree[pos], '\n')

    # print
    # "Sentence 3:"
    # print
    # tree3
    # print
    # "Sentence 4:"
    # print
    # tree4
    # tree, pos = hobbs([tree3, tree4], (1, 1, 0))
    # print
    # "Proposed antecedent for 'it':", tree[pos]
    # tree, pos = hobbs([tree3, tree4], (0, 0))
    # print
    # "Proposed antecedent for 'he':", tree[pos], '\n'
    #
    # print
    # "Sentence 5:"
    # print
    # tree5
    # tree, pos = hobbs([tree5], (1, 2, 1, 1, 0, 0))
    # print
    # "Proposed antecedent for 'he':", tree[pos], '\n'
    #
    # print
    # "Sentence 6:"
    # print
    # tree6
    # tree, pos = resolve_reflexive([tree6], (1, 1, 1, 1, 1, 0))
    # print
    # "Proposed antecedent for 'herself':", tree[pos], '\n'


def main(argv):
    if len(sys.argv) == 2 and argv[1] == "demo":
        demo()
    else:
        if len(sys.argv) > 3 or len(sys.argv) < 2:
            print("Enter the file and the pronoun to resolve.")
        elif len(sys.argv) == 3:
            p = ["He", "he", "Him", "him", "She", "she", "Her",
                 "her", "It", "it", "They", "they"]
            r = ["Himself", "himself", "Herself", "herself",
                 "Itself", "itself", "Themselves", "themselves"]
            fname = sys.argv[1]
            pro = sys.argv[2]
            with open(fname) as f:
                sents = f.readlines()
            trees = [Tree.fromstring(s) for s in sents]
            pos = get_pos(trees[-1], pro)
            pos = pos[:-1]
            if pro in p:
                tree, pos = hobbs(trees, pos)
                for t in trees:
                    print(t, '\n')
                print("Proposed antecedent for '" + pro + "':", tree[pos])
            elif pro in r:
                tree, pos = resolve_reflexive(trees, pos)
                for t in trees:
                    print(t, '\n')
                print("Proposed antecedent for '" + pro + "':", tree[pos])


if __name__ == "__main__":
    main(sys.argv)
