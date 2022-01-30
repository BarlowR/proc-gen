import random
import numpy as np
import matplotlib.pyplot as plt


within_bounds = lambda pos, prev_pos, env : \
               (pos[0] >= 0 and
                pos[1] >= 0 and
                pos[0] <= env.shape[0]-1 and
                pos[1] <= env.shape[1]-1)

def euclidean_distance(p, q):
    return np.linalg.norm(np.asarray(p) - np.asarray(q))

def step_distace(p, q):
    return (np.abs(p[0]-q[0]) + np.abs(p[1]-q[1]))


def a_star( env, 
            start, 
            end, 
            passable = within_bounds,
            distance_heuristic = euclidean_distance):

    #node queue tuple: (pos, prev_pos, f, d) where f is d + h, d = absolute distance from start, h is distance to end heuristic)
    node_queue = [(  start,
                    start,
                    0 + 2*distance_heuristic(start, end), 
                    0)]

    #visited node dict: pos : prev_pos
    visited_nodes = {}

    current_node = node_queue.pop(0)

    while (current_node[0] != end):
        #print(current_node[0])
        #print(len(visited_nodes))
        current_pos = current_node[0]
        prev_pos = current_node[1]
        current_d = current_node[3]
        for (x,y) in [(-1,0), (1,0), (0,-1), (0,1)]:

            new_pos = (current_pos[0]+x, current_pos[1]+y)

            if (    passable(new_pos, current_pos, env) and 
                    not (new_pos in visited_nodes.keys())):
                new_d = distance_heuristic(new_pos, start)
                new_h = distance_heuristic(new_pos, end)
                new_f = new_d+new_h

                new_node = (new_pos, current_pos, new_f, new_d)
                ##print(visited_nodes.keys())
                queue_nodes = [node for node in node_queue if node[0] == new_node[0]]
                for old_queue_node in queue_nodes:
                    if old_queue_node[2] < new_node[2]: continue 
                
                if (new_node in node_queue): continue

                add_inplace_sorted_f(new_node, node_queue)

        visited_nodes[current_pos] = prev_pos 
        if (len(node_queue) > 0): current_node = node_queue.pop(0)
        else: return []

        ##print(visited_nodes.keys())
        

    visited_nodes[end] = current_node[1]
    return link_path(visited_nodes, start, end)



##
def add_inplace_sorted_f(node, node_list):
    i = 0
    while (i < len(node_list)):
        if node[2] < node_list[i][2]:
            node_list.insert(i, node)
            return
        i+=1
    
    node_list.insert(len(node_list), node)

def test_add_inplace():
    list = [    (0,0,0,0), 
                (0,0,2,0), 
                (0,0,6,0), 
                (0,0,9,0), 
                (0,0,9,0)]
    new_node =  (0,0,7,0)

    add_inplace_sorted_f(new_node, list)

    if (list ==[(0,0,0,0), 
                (0,0,2,0), 
                (0,0,6,0), 
                (0,0,7,0), 
                (0,0,9,0),
                (0,0,9,0)]): print("working")
    else: print("broken")

def link_path(node_dict, start, end):
    path = [end]

    while (path[0] != start):
        path.insert(0, node_dict[path[0]])
    return path


if __name__ == "__main__":
    #test_add_inplace()

    env = np.random.rand(40, 200)

    start = (1,1)
    end = (20, 90)
    path = a_star(env, start, end, passable = lambda pos, prev_pos, env :\
                                    within_bounds(pos, prev_pos, env) and
                                    env[pos[0], pos[1]] < 0.65)
    for node in path:
        env[node[0], node[1]] = 1
    plt.imshow(env)
    plt.show()