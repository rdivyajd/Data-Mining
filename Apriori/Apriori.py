#!/usr/bin/env python3
# Divya Rajendran 2018
# I have taken the data sets from UCI machine learning repository. http://archive.ics.uci.edu/ml/datasets/
# All the data is saved into a csv, one data set has a duplicate column and had to be removed.

# Data Sets:
# Below are the Data Sets in order taken from UCI Machine Learning Repository
# 1. Tic-Tac-Toe: http://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame - Number of records - 958
# 2. Nursery: http://archive.ics.uci.edu/ml/datasets/Nursery - Number of records - 9573 -
# considered 8 attributes as mentioned in the DS data set contained 9th column duplicating the 8th column,
# removed it from the data
# 3. Condition Based Maintenance of Naval Propulsion Plants Data Set:
# http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
# - Number of records - 11934

# Executing the program:
# python ./Q2.py File_Name minimum_support minimum_confidence_or_lift method measure
# File_Names: Name of the file, takes the values ["Tac-Toe.csv", "Nursery.csv", "plant.csv"]
# Values considered for minimum_support: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# Values considered for minimum_confidence_or_lift: [0.5, 0.6, 0.7]
# Values for method: ["Fk-1XFk-1", "Fk-1XF1", "bruteforce"]
# Values for measure: ["confidence", "lift"]

# Algorithm: Apriori Algorithm
# Program Flow:
# Each function has a comment on top of it indicating its purpose
# First the Data set is read into a variable, post which the data is turned into a binary format through the function
# "convert_binary", here code is incorporated to restrict the number of items in each of the columns if its greater than
#  3 times the columns in the data set.
# Once we have the binary transaction data, we feed the data into "apriori_algorithm", where apriori algorithm processes
# Output of the "apriori_algorithm" function is frequent item sets, association rules, total candidate items generated
# and the total number of frequent items
# Each of the questions in the Assignment is answered in the form of print statements with appropriate indication
# of the sub question.

# used to get the arguments from command line in terminal
import sys

# read the csv row by row
def read_csv(filename):
    f = open(filename, 'rU')
    for l in f:
        row = frozenset(l.strip().rstrip(',').split(','))
        yield row


# median of a list
def median(lst):
    n = len(lst)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(lst)[n//2]
    else:
            return sum(sorted(lst)[n//2-1:n//2+1])/2.0


# conversion of our data set into binary format
def convert_binary(item_set, transactions_list, greater):
    len_unique = len(item_set)
    binary_transactions = list()
    new_transaction_list = list()
    updated_transactions_list = []
    if greater == 1:
        column_max_mins = []
        converted_list = list()
        for row in transactions_list:
            row_append = []
            for item in list(frozenset(row)):
                if item != 'ï»¿1.14E+00':
                    row_append.append(float(item))
                else:
                    row_append.append(0)
            converted_list.append(row_append)

        columns = len(converted_list[0])
        for itr in range(columns):
            column = list()
            for row in converted_list:
                if len(row) <= itr:
                    column.append(0)
                else:
                    column.append(row[itr])
            # column = [row[itr] for row in converted_list]
            median_value = median(column)
            min_value = min(column)
            max_value = max(column)
            column_max_mins.append([min_value, median_value, max_value])

        itr = 0
        new_transaction_list = [[ col for col in range(columns)] for row in converted_list]
        for row in converted_list:
            for col in range(columns):
                if len(row) <= col:
                    row.append(0)
                if row[col] <= column_max_mins[col][0]:
                    new_transaction_list[itr][col] = column_max_mins[col][0]
                elif column_max_mins[col][0] < row[col] <= column_max_mins[col][1]:
                    new_transaction_list[itr][col] = column_max_mins[col][1]
                elif column_max_mins[col][1] < row[col] <= column_max_mins[col][2]:
                    new_transaction_list[itr][col] = column_max_mins[col][2]
            itr += 1

        new_items_set = set()
        for row in new_transaction_list:
            updated_transactions_list.append(frozenset(row))
            for item in frozenset(row):
                if item:
                    new_items_set.add(frozenset([item]))
    if len(updated_transactions_list) > 0:
        transactions = updated_transactions_list
        items_set = new_items_set
    else:
        transactions = transactions_list
        items_set = item_set

    for row in transactions:
        binary_values = [None for i in range(len_unique)]
        itr = 0
        for item in items_set:
            if item.issubset(frozenset(row)):
                binary_values[itr] = 1
            itr += 1
        binary_transactions.append(binary_values)

    return binary_transactions


# converting binary transactions to the format we need
def binary_to_items(item_set, transactions_list):
    items = list()
    for item in item_set:
        sample = list(item)
        for value in sample:
            items.append(value)

    normal_transactions = list()
    for transaction in transactions_list:
        normal = list()
        for itr in range(len(items)):
            if transaction[itr] == 1:
                normal.append(items[itr])
        normal_transactions.append(frozenset(normal))

    return normal_transactions


# read the data from the csv saved in a generator object # to get 1 item sets
def items_set_from_data(data):
    items_set = set()
    transactions_list = list()
    sample_set = set()
    greater = 0
    count = 0
    for row in data:
        count += 1
        transactions_list.append(frozenset(row))
        for item in frozenset(row):
            if item:
                items_set.add(frozenset([item]))
                sample_set.add(item)
    columns_count = len(transactions_list[0])
    if len(items_set) > 3 * columns_count and count != 9573:
        # print("item set count ", len(items_set), "columns count ", columns_count)
        greater = 1

    return items_set, transactions_list, greater


# creation of candidate sets
# brute force method
def brute_force(items_set, length):
    combinations = combination(items_set, length)
    all_sets = set()
    for set_val in combinations:
        all_values =list()
        # print(set_val)
        for value in set_val:
            value1 = list(value)
            all_values.extend(value1)
        all_sets.add(frozenset(all_values))
    print("done brute force")
    return all_sets


# creation of candidate sets
# F_k-1 x F_k-1 method
def candidate_gen_fk_1_fk_1(item_set, length):
    return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])


# creation of candidate sets
# F_k-1 x F_1 method
def candidate_gen_fk_1_f1(item_set, length, all_items):
    sample_set = list()
    # set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])
    for i in item_set:
        for j in all_items:
            if len(i.union(j)) == length:
                sample_set.append(i.union(j))
    return set(sample_set)

    # return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])


# default dictionary definition
def default_dict(default_type):
    class DefaultDict(dict):
        def __getitem__(self, key):
            if key not in self:
                dict.__setitem__(self, key, default_type())
            return dict.__getitem__(self, key)
    return DefaultDict()


# splits and combines an element into combination of elements ('ABCD', 2) --> AB AC AD BC BD CD
# python 3 documentation for itertools.combinations equivalent
def combination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))

    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] = indices[i] + 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


# converts list of elements into individuals ['ABC', 'DEF'] --> A B C D E F
# python 3 documentation for itertools.chain equivalent
def from_iterable(iterables):
    for it in iterables:
        for element in it:
            yield element


# returns a subset of an array
def subsets(array):
    return from_iterable([combination(array, i + 1) for i, a in enumerate(array)])


# find item sets with minimum support - also calculations of support for items
def items_with_min_support(items_set, transactions_list, minimum_support, frequent_set):
    item_set = set()
    local_set = default_dict(int)

    for item in items_set:
        for transaction in transactions_list:
            if item.issubset(transaction):
                frequent_set[item] += 1
                local_set[item] += 1

    for item, count in local_set.items():
        support = float(count)/len(transactions_list)

        if support >= minimum_support:
            item_set.add(item)

    return item_set


# item's support
def item_support(frequent_set, transactions_list, item):
    return float(frequent_set[item])/len(transactions_list)


# all sets in the data, after converting binary transactions to the format we need
def all_sets(item_set, transactions, minimum_support, method):
    # transactions_list = transactions
    transactions_list = binary_to_items(item_set, transactions)
    frequent_set = default_dict(int)
    all_set = dict()
    first_set = items_with_min_support(item_set, transactions_list, minimum_support, frequent_set)
    last_set = first_set
    # print(last_set, "\n")

    k = 2
    total_candidates = len(last_set)
    while last_set != set([]):
        all_set[k-1] = last_set

        if method == "Fk-1XFk-1":
            last_set = candidate_gen_fk_1_fk_1(last_set, k)
        elif method == "Fk-1XF1":
            last_set = candidate_gen_fk_1_f1(last_set, k, item_set)

        # count the total number of candidate sets
        total_candidates += len(last_set)
        # len(last_set) -- to count the number of generated candidate sets
        current_set = items_with_min_support(last_set, transactions_list, minimum_support, frequent_set)
        last_set = current_set
        k = k + 1
    return all_set, frequent_set, transactions_list, total_candidates


# calculate frequent maximal, frequent closed and closed item sets
def frequent_maximal_closed(frequent_items):

    level_freq_dic = dict()
    for item in frequent_items:
        key = len(item[0])
        if key not in level_freq_dic.keys():
            level_freq_dic[key] = list()
        item_new = list(item)
        level_freq_dic[key].append([item_new[0], item_new[1]])
    # print("level_freq_dic", level_freq_dic)

    maximal_frequent = []
    closed = []
    for key in level_freq_dic.keys():
        # get the maximal frequent sets list
        for set_val in level_freq_dic[key]:
            if key+1 in level_freq_dic.keys() and set_val[0][key-1] in [value[0][key-1] for value in level_freq_dic[key + 1]]:
                continue
            else:
                maximal_frequent.append(set_val)
        # get the closed frequent sets list
        for set_val in level_freq_dic[key]:
            if key+1 in level_freq_dic.keys() and set_val[1] in [value[1] for value in level_freq_dic[key + 1]]:
                continue
            else:
                closed.append(set_val)
    # print("maximum frequent :", maximal_frequent)
    # print("closed set is: ", closed)

    # get the closed frequent set lists
    closed_frequent = []
    for set_val in closed:
        if set_val[1] >= minimum_support:
            closed_frequent.append(set_val)
    # print("closed frequent is: ", closed_frequent)

    return maximal_frequent, closed_frequent


# Apriori Algorithm implementation
def apriori_algorithm(item_set, binary_transactions, minimum_support, minimum_confidence_or_lift, method, measure):
    # all_set, frequent_set, transactions_list, total_candidates = \
    #     all_sets(item_set, binary_transactions, minimum_support, method)

    transactions_list = binary_to_items(item_set, binary_transactions)
    frequent_set = default_dict(int)
    all_set = dict()
    first_set = items_with_min_support(item_set, transactions_list, minimum_support, frequent_set)
    last_set = first_set

    # print("Finished 1")

    k = 2
    total_candidates = len(last_set)

    # Candidate items generation
    while last_set != set([]):
        all_set[k - 1] = last_set
        if method == "Fk-1XFk-1":
            last_set = candidate_gen_fk_1_fk_1(last_set, k)
        elif method == "Fk-1XF1":
            last_set = candidate_gen_fk_1_f1(last_set, k, item_set)
        elif method == "bruteforce":
            last_set = brute_force(item_set, k)

        # count the total number of candidate sets
        total_candidates += len(last_set)
        # len(last_set) -- to count the number of generated candidate sets
        current_set = items_with_min_support(last_set, transactions_list, minimum_support, frequent_set)
        last_set = current_set
        k = k + 1

    # print("Finished 2")

    # Frequent Items
    frequent_items = []
    for key, value in all_set.items():
        frequent_items.extend([(tuple(item), item_support(frequent_set, transactions_list, item))
                           for item in value])
    total_frequent_items = len(frequent_items)

    # print("all sets are: ", all_set)
    # print("\nfrequent items are: ", frequent_items)

    association_rules = []
    for key, value in all_set.items():
        for item in value:
            sub_sets = map(frozenset, [value_set for value_set in subsets(item)])
            for element in sub_sets:
                remainder = item.difference(element)
                if len(remainder) > 0:
                    confidence = item_support(frequent_set, transactions_list, item) / \
                                 item_support(frequent_set, transactions_list, element)
                    lift = confidence/minimum_support
                    if measure == "confidence" and confidence >= minimum_confidence_or_lift:
                        association_rules.append(((tuple(element), tuple(remainder)), confidence))
                    elif measure == "lift" and lift >= minimum_confidence_or_lift:
                        association_rules.append(((tuple(element), tuple(remainder)), lift))

    return frequent_items, association_rules, total_candidates, total_frequent_items


# Main Program Starts
if __name__ == "__main__":
    file_name = sys.argv[1]

    minimum_support = float(sys.argv[2])
    minimum_confidence_or_lift = float(sys.argv[3])
    method = sys.argv[4]
    measure = sys.argv[5]

    data_file = read_csv(file_name)  # ["Tic-Tac-Toe.csv", "Nursery.csv", "plant.csv"] #  # sys.argv[1]
    # read the data from the file
    item_set, transactions_list, greater = items_set_from_data(data_file)
    # convert data into binary transactions
    binary_transactions = convert_binary(item_set, transactions_list, greater)
    # Apriori Algorithm Call
    frequent_items, association_rules, total_candidates, total_frequent_items = \
        apriori_algorithm(item_set, binary_transactions, minimum_support, minimum_confidence_or_lift, method, measure)
    # Maximal and closed frequent item sets calculations
    maximal_frequent, closed_frequent = frequent_maximal_closed(frequent_items)


    print("\n------------------------ Question A & B -----------------------------:")
    print("The number of candidate sets generated for the method ", method, "is: ", total_candidates)
    print("\nThe number of frequent item sets generated is: ", total_frequent_items)
    print("-------------------------------------------------------------------")

    print("\n------------------------ Question C -----------------------------:")
    print("The number of frequent closed item sets for the method ", method, "is: ", len(closed_frequent))
    print("\nThe number of maximal frequent item sets is: ", len(maximal_frequent))
    print("-------------------------------------------------------------------")

    print("\n------------------------ Question D -----------------------------:")
    print("The number of generated confidence rules for the method ", method, "is: ", len(association_rules))
    print("-------------------------------------------------------------------")

    if measure == "confidence":
        print("\n------------------------ Question E -----------------------------:")
        print("\nThe top five association rules using confidence, for the method ", method, "are below: \n")
        for rule, confidence in \
                sorted(association_rules, key=lambda p: (lambda rule, confidence: confidence)(*p), reverse=True)[0:5]:
            pre, post = rule
            print("Rule: %s ==> %s with confidence, %.3f" % (str(pre), str(post), confidence))
        print("-------------------------------------------------------------------")
    elif measure == "lift":
        print("\n------------------------ Question F -----------------------------:")
        print("\nThe top five association rules using lift, for the method ", method, "are below: \n")
        for rule, lift in \
                sorted(association_rules, key=lambda p: (lambda rule, lift: lift)(*p), reverse=True)[0:5]:
            pre, post = rule
            print("Rule: %s ==> %s with confidence, %.3f" % (str(pre), str(post), lift))
        print("-------------------------------------------------------------------")