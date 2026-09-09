import numpy as np
from typing import Optional, Union, List

# Every mechanism's crank is nodes 0 and 1: always fixed at this unit-length position,
# always linked to each other, with node 0 always a fixed ground pivot, node 1 never
# fixed, and the motor always driving the (0, 1) link. This is the ONLY place these
# facts should be written; nothing outside LINKS should ever construct or assume them.
CRANK = np.array([[0.0, 0.0], [0.0, 1.0]])
CRANK_EDGE = np.array([[0, 1]])
MOTOR = np.array([0, 1])

def with_crank(x0: np.ndarray) -> np.ndarray:
    '''
    Prepend the fixed crank (nodes 0 and 1) to a mechanism's free joint positions
    (nodes 2 onward). Every public LINKS function that takes joint positions accepts
    only the free positions and calls this internally -- callers never provide, see,
    or handle the crank's position.
    '''
    return np.vstack([CRANK, x0])

def with_crank_edges(edges: np.ndarray) -> np.ndarray:
    '''
    Prepend the fixed crank link (0, 1) to a mechanism's other edges. Every public
    LINKS function that takes edges accepts only the non-crank edges and calls this
    internally -- callers never provide, see, or handle the crank's own link.
    '''
    edges = np.asarray(edges)
    if edges.size == 0:
        edges = edges.reshape(0, 2)
    return np.vstack([CRANK_EDGE, edges]).astype(int)

def with_crank_fixed(fixed_joints: np.ndarray) -> np.ndarray:
    '''
    Prepend node 0 (the crank's fixed ground pivot) to a mechanism's other fixed
    joints. Node 1 (the crank tip) is never fixed, so it never needs to be excluded.
    Every public LINKS function that takes fixed joints accepts only the non-crank
    fixed joints and calls this internally -- callers never provide node 0.
    '''
    fixed_joints = np.asarray(fixed_joints).reshape(-1)
    return np.concatenate([[0], fixed_joints]).astype(int)

def dyadic_path(
    edges : np.ndarray,
    fixed_nodes : np.ndarray,
    N : int,
    motor : Optional[Union[np.ndarray, List[int]]] = [0, 1]
):
    knowns = np.append(fixed_nodes, motor[1])
    unkowns = np.arange(N)
    unkowns = unkowns[~np.isin(unkowns, knowns)]
    
    full_edges = np.concatenate((edges, np.fliplr(edges)))
    full_edges = np.unique(full_edges, axis=0)
    full_edges = full_edges[np.argsort(full_edges[:,0])]
    
    ptr = np.searchsorted(full_edges[:,0], np.arange(N))
    ptr = np.append(ptr, full_edges.shape[0])
    
    counter = 0
    path = np.zeros((unkowns.shape[0], 3), dtype=edges.dtype)
    pc = 0
    while unkowns.shape[0] != 0:
        if counter == unkowns.shape[0]:
            # Non dyadic or DOF larger than 1
            return path, 1
        
        n = unkowns[counter]
        ne = full_edges[ptr[n]:ptr[n+1], 1]

        kne = ne[np.isin(ne,knowns)]
        
        if kne.shape[0] == 2:
            path[pc, 0] = n
            path[pc, 1] = kne[0]
            path[pc, 2] = kne[1]
            pc += 1
            knowns = np.insert(knowns, 0, n)
            unkowns = np.delete(unkowns, counter)
            counter = 0
        elif kne.shape[0] > 2:
            #redundant or overconstraint
            return path, 2
        else:
            counter += 1

    return path, 0

def sort_mechanism(
    x0 : np.ndarray,
    edges : np.ndarray,
    fixed_nodes : np.ndarray,
    N : int,
    motor : Optional[Union[np.ndarray, List[int]]] = [0, 1]
):
    path, status = dyadic_path(edges, fixed_nodes, N, motor)
    
    if status != 0:
        if status == 1:
            raise ValueError("Mechanism is not dyadic or has DOF larger than 1")
        elif status == 2:
            raise ValueError(f"Mechanism has redundant linkages or is overconstrained")

    order = np.append(
        np.insert(
            fixed_nodes[fixed_nodes != motor[0]],
            0,
            motor
        ),
        path[:, 0]
    )
    
    mapping = np.argsort(order)
    
    fixed_nodes_sorted = mapping[fixed_nodes]
    edges_sorted = np.zeros_like(edges)
    edges_sorted[:, 0] = mapping[edges[:, 0]]
    edges_sorted[:, 1] = mapping[edges[:, 1]]
    x0_sorted = x0[order].squeeze()
    

    return edges_sorted, fixed_nodes_sorted, N, np.array([0,1]), x0_sorted, order, mapping