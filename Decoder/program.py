import string
import heapq
from queue import Queue
import math
import heapq
import sys

class Node:
    def __init__(self, string, swap_key, visited ,swapped_keys, threshold, cost):
        self.string = string
        self.swap_key = swap_key
        self.visited = visited
        self.swapped_keys = swapped_keys
        self.threshold = threshold
        self.cost = cost
        self.children = []
        
    def __lt__(self, other):
        # Compare nodes based on f value (g + h)
        return None
    def add_child(self, child):
        self.children.append(child)
  

class Tree:
    def __init__(self):
        self.root = None

    def add_root(self, node):
        if self.root == None:
            self.root = node
        else:
            print("Root already exists\n")



def split_message(message): 
    dictionary = []
    translator = str.maketrans('', '', string.punctuation)

    msg = message.split()

    for word in msg:
        dictionary.append(word.strip().translate(translator).lower())
    
    return dictionary

def load_message_whole(filename):
    with open(filename, 'r') as file:
        return file.read()

def load_dict(filename):
    dictionary = set()
    with open(filename, 'r') as file:
        for line in file:
            dictionary.add(line.strip().lower())
      
    return dictionary

def threshold_f(message, dictionary, threshold_i):
    #TODO
    result = ''
    threshold = 0
    count = sum(1 for word in message if word in dictionary)
    if count > 0:
        float = count/len(message)

        threshold = float * 100

    if threshold_i <= threshold:
        result += "True\n"
    else: 
        result+= "False\n"
    result +="{:.2f}".format(threshold)

    return result


def possible_swaps(s):
    swaps = []
    for i, letter1 in enumerate(s):
        for j, letter2 in enumerate(s):
            if i != j and letter1 != letter2:
                if (letter1, letter2) not in swaps and (letter2, letter1) not in swaps:
                    if letter1 < letter2:
                        swaps.append((letter1, letter2))
                    else:
                        swaps.append((letter2, letter1))

    return sorted(swaps, key=lambda x: (x[0], x[1]))

def perform_swap(key, s):
    final_con = ''

    content = s
  
    first_con = content.replace(key[0].lower(), '#')
    
    # print(char[0], char[1])
    # print(first_con)
    second_con = first_con.replace(key[1].lower(), key[0].lower())
    # print(second_con)
    final_con = second_con.replace('#', key[1].lower())
    # print(final_con)
    
    first_con = final_con.replace(key[0], '#')
    # print(first_con)
    second_con = first_con.replace(key[1], key[0])
    # print(second_con)
    final_con = second_con.replace('#', key[1])
    
    return final_con


def generate_children(swap_keys, current_node, dictionary, threshold_i):
    #add childrens
    if len(current_node.children) == 0:
        for swap in swap_keys:
            #made node and add child to curr node
            child_message = perform_swap(swap, current_node.string)
            child = Node(child_message,swap_keys, None, current_node.swapped_keys + swap[0] + swap[1],threshold_f(split_message(child_message), dictionary,threshold_i),current_node.cost + 1)
            current_node.add_child(child)
    
    return current_node
    
   

def DFS(root, debug, count, swap_keys, dictionary, threshold_i):
    stack = [(root, 0)]

    visited = []
    result = ''
    expanded = 0
    fringe = 0

    nodes_expanded = "First few expanded states:\n"
   
    while stack:
        if len(stack) > fringe:
            fringe = len(stack)

        if expanded < 1000:
            current_node, depth = stack.pop()
            # print("depth ={}".format(depth))
        
    
        if expanded == 1000:
            result += "No solution found.\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(depth)
            if debug == 'y':
                result += '\n\n' + nodes_expanded
            return result

        
        expanded += 1
      
    
        if "True" in current_node.threshold:
            nodes_expanded += current_node.string
            result += "Solution: {}\n\nKey: {}\nPath Cost: {}\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(depth)
            if debug == 'y':
                result += "\n\n" + nodes_expanded
            return result
                

        if expanded < 10:
            nodes_expanded += current_node.string + "\n\n"
        elif expanded == 10:
            nodes_expanded += current_node.string
  
    
        generate_children(swap_keys,current_node,dictionary,threshold_i)
        children = current_node.children[::-1]
        
        
        for child in children:
            if child:
                # print("HERE")
                stack.append((child, depth + 1))
    return result


def BFS(root, debug, count, swap_keys, dictionary, threshold_i):
    queue = [root]
    visited = set()
    result = ''
    expanded = 0
    fringe = 0
    depth = 0
    nodes_expanded = "First few expanded states:\n"
    while queue:
        if len(queue) > fringe:
            fringe = len(queue)
        if expanded < 1000:
            current_node = queue.pop(0)
     
        visited.add(current_node)

                
        if expanded == 1000:
            result += "No solution found.\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(depth)
            if debug == 'y':
                result += "\n\n" + nodes_expanded
            return result

        expanded += 1
        if "True" in current_node.threshold:
            nodes_expanded += current_node.string
            result += "Solution: {}\n\nKey: {}\nPath Cost: {}\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(current_node.cost)
            if debug == 'y':
                result += "\n\n" + nodes_expanded
            return result
        
        if expanded < 10:
            nodes_expanded += current_node.string + "\n\n"
        elif expanded == 10:
            nodes_expanded += current_node.string

        
        generate_children(swap_keys,current_node,dictionary,threshold_i)


        for child in current_node.children:
            if child not in visited:
                queue.append(child)
                visited.add(child)



    return "Solution not found: \n" + nodes_expanded

def IDS(root, debug, count, swap_keys, dictionary, threshold_i):
    depth_limit = 0
    result = ''
    expanded = 0
    fringe = 0
    depth = 0
    nodes_expanded = "First few expanded states:\n"
    while True:
        visited = set()
        stack = [(root, 0)]
        while stack:
            if len(stack) > fringe:
                fringe = len(stack)
            if expanded < 1000:
                current_node, depth = stack.pop(0)

            
            # print("current = {} swappedkeys = {}".format(current_node.string, current_node.swapped_keys))
            # print("depth_limit = {}".format(depth_limit))
            if expanded == 1000:
                result += "No solution found.\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
                result += "Num nodes expanded: {}\n".format(expanded)
                result += "Max fringe size: {}\n".format(fringe)
                result += "Max depth: {}".format(depth_limit)
                if debug == 'y':
                    result += '\n\n' + nodes_expanded
                return result

            expanded += 1

            if "True" in current_node.threshold:

                nodes_expanded += current_node.string

                result += "Solution: {}\n\nKey: {}\nPath Cost: {}\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
                result += "Num nodes expanded: {}\n".format(expanded)
                result += "Max fringe size: {}\n".format(fringe)
                result += "Max depth: {}".format(depth_limit)
                if debug == 'y':
                    result += '\n\n' + nodes_expanded
                return result

              
            if expanded < 10:
                nodes_expanded += current_node.string + "\n\n"
            elif expanded == 10:
                nodes_expanded += current_node.string
            
            generate_children(swap_keys,current_node,dictionary,threshold_i)

            if depth < depth_limit:
                visited.add(current_node)
                children = current_node.children[::-1]
                for child in children:
                    if child not in visited:
                        if depth < depth_limit:
                            stack.insert(0,(child, depth + 1))
                        
                            # stack.append((child, depth + 1))

                            
        depth_limit += 1
    

def UCS(root, debug, count, swap_keys, dictionary, threshold_i):
    queue = Queue()
    queue.put((0, [root]))
    visited = set() 
    result = ''
    expanded = 0
    fringe = 0
    nodes_expanded = "First few expanded states:\n"
    while not queue.empty():
        if queue.qsize() > fringe:
            fringe = queue.qsize()
        if expanded < 1000:
            path_cost, path = queue.get()
        current_node = path[-1]
        
        if expanded == 1000:
                result += "No solution found.\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
                result += "Num nodes expanded: {}\n".format(expanded)
                result += "Max fringe size: {}\n".format(fringe)
                result += "Max depth: {}".format(path_cost)
                if debug == 'y':
                    result += '\n\n' + nodes_expanded
                return result

        
        expanded += 1
        if "True" in current_node.threshold:
            nodes_expanded += current_node.string
            result += "Solution: {}\n\nKey: {}\nPath Cost: {}\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(current_node.cost)
            
            if debug == 'y':
                result+= '\n\n' + nodes_expanded
            return result
        
                  
        if expanded < 10:
            nodes_expanded += current_node.string + "\n\n"
        elif expanded == 10:
            nodes_expanded += current_node.string
        if current_node in visited:
            continue
        
        # print("NODE str = {} NODE cost = {} swapped keys = {}".format(current_node.string, current_node.cost, current_node.swapped_keys ))
     
        generate_children(swap_keys, current_node, dictionary, threshold_i)
        
        
        for child in current_node.children:
            child_path = list(path)
            child_path.append(child)
            child_path_cost = path_cost + 1
            queue.put((child_path_cost, child_path))
    return result

def heuristic(message, threshold):

    # Define the goal ordering and the set of relevant letters
    th_goal = "ETAONS"
    check_letters = set("ETAONS")
    is_goal = False
    if "True" in threshold:
        is_goal = True

    # Read in the message from the file

    # Count the occurrence of relevant letters in the message
    counts = []
    for letter in check_letters:
        counts.append((letter, message.lower().count(letter.lower())))
   

    sorted_counts = sorted(counts, key=lambda x: (-x[1], x[0])) # negative x[1] to sort by descending

    sorted_string = ''

    for e in sorted_counts:
        sorted_string += e[0]
    # print(sorted_string)

    goal_count = 0
    for i in range(len(th_goal)):
        if th_goal[i] != sorted_string[i]:
            goal_count += 1
        
    # print("sorted = {} th_goal = {}".format(sorted_string, th_goal))
          

    if is_goal:
        return 0
    else:
        return math.ceil(goal_count / 2)
    

def GREEDY(root, debug, count, swap_keys, dictionary, threshold_i):
    queue = [(heuristic(root.string, root.threshold), root)]
    result = ''
    expanded = 0
    fringe = 0
    nodes_expanded = "First few expanded states:\n"

    while queue:
        if len(queue) > fringe:
            fringe = len(queue)
        if expanded < 1000:
            h, current_node = heapq.heappop(queue)

        if expanded == 1000:
            result += "No solution found.\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(current_node.cost)
            if debug == 'y':
                result += '\n\n' + nodes_expanded
            return result

        
        expanded += 1

        if "True" in current_node.threshold:
            nodes_expanded += current_node.string
            result += "Solution: {}\n\nKey: {}\nPath Cost: {}\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(current_node.cost)
            
            if debug == 'y':
                result+= '\n\n' + nodes_expanded
            return result
        
        
        if expanded < 10:
            nodes_expanded += current_node.string + "\n\n"
        elif expanded == 10:
            nodes_expanded += current_node.string

        generate_children(swap_keys, current_node, dictionary, threshold_i)

        for child in current_node.children:
            heapq.heappush(queue, ((heuristic(child.string, child.threshold)), child))


def A_STAR(root, debug, count, swap_keys, dictionary, threshold_i):
    open_set = [(heuristic(root.string, root.threshold), 0, root)]
    closed_set = set()
    result = ''
    expanded = 0
    fringe = 0
    nodes_expanded = "First few expanded states:\n"

    while open_set:
        if len(open_set) > fringe:
            fringe = len(open_set)

        if expanded < 1000:
            h, path_cost, current_node = heapq.heappop(open_set)
        
        if expanded == 1000:
            result += "No solution found.\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(current_node.cost)

            if debug == 'y':
                result += '\n\n' + nodes_expanded
            return result


        if "True" in current_node.threshold:
            nodes_expanded += current_node.string
            result += "Solution: {}\n\nKey: {}\nPath Cost: {}\n\n".format(current_node.string, current_node.swapped_keys, current_node.cost)
            result += "Num nodes expanded: {}\n".format(expanded)
            result += "Max fringe size: {}\n".format(fringe)
            result += "Max depth: {}".format(current_node.cost)

            if debug == 'y':
                result += '\n\n' + nodes_expanded
            return result

       

        expanded += 1

        if expanded < 10:
            nodes_expanded += current_node.string + "\n\n"
        elif expanded == 10:
            nodes_expanded += current_node.string

        generate_children(swap_keys, current_node, dictionary, threshold_i)

        for child in current_node.children:
            if child in closed_set:
                continue

            child_cost = path_cost + child.cost
            f = child_cost + heuristic(child.string, child.threshold)

            if child in [n[2] for n in open_set]:
                index = [n[2] for n in open_set].index(child)
                if open_set[index][0] > f:
                    open_set[index] = (f, child_cost, child)
                    child.parent = current_node
                    child.cost = child_cost
            else:
                heapq.heappush(open_set, (f, child_cost, child))
                child.parent = current_node
                child.cost = child_cost

        closed_set.add(current_node)

def task6(algorithm, message_filename, dictionary_filename, threshold, letters, debug):

    #TODO
    swap_keys = possible_swaps(letters)
    # print("swap keys = {}".format(swap_keys))
    message_whole = load_message_whole(message_filename)
  
    # print(message)
    dictionary = load_dict(dictionary_filename)
    threshold_i = threshold

    
    tree = Tree()
    root = Node(message_whole,swap_keys, None, '', threshold_f(split_message(message_whole),dictionary,threshold_i), 0)
    
    tree.add_root(root)
    
    result = ''
 
    if algorithm == "d":
        result = DFS(tree.root, debug,0,swap_keys,dictionary,threshold_i)
    elif algorithm == "b":
        result = BFS(tree.root, debug,0,swap_keys,dictionary,threshold_i)
    elif algorithm == "i":
        result = IDS(tree.root, debug,0,swap_keys,dictionary,threshold_i)
     
    elif algorithm == "u":
        result = UCS(tree.root, debug,0,swap_keys,dictionary,threshold_i)

    elif algorithm == "g":
        result = GREEDY(tree.root, debug,0,swap_keys,dictionary,threshold_i)

    elif algorithm == "a":
        result = A_STAR(tree.root, debug,0,swap_keys,dictionary,threshold_i)

    else:
        print("""Input error: Please specify the correct algorithm using letters in the command line:
    -> d - DFS
    -> b - BFS
    -> i - IDS
    -> u - UCS
    -> g - Greedy
    -> a - A*""")
    return result

if __name__ == '__main__':
    try: 
        algorithm = sys.argv[1]
        print(task6(algorithm, 'secret_msg.txt', 'common_words.txt', 90, 'AENOST', 'n'))
        print("\n")
        print(task6(algorithm, 'cabs.txt', 'common_words.txt', 90, 'ABC', 'y'))
    except:
         print("""Input error: Please specify the correct algorithm using letters in the command line:
    -> d - DFS
    -> b - BFS
    -> i - IDS
    -> u - UCS
    -> g - Greedy
    -> a - A*""")