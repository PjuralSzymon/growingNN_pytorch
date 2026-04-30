
In this module we have functions used by actions to analyse the graph where to add or delete what node and what is the model strucuture but there is no funcitons tahta re cahning the graph

---
 For adding layers we have the following 2 functions:

  
### `module_dependency_pairs` 
Returns all **reachable module pairs** `(ancestor, descendant)`.  
Used in adding: [[Residual actions]]
  
- Captures **full dependency paths** (transitive connections)  
- Includes indirect relationships  

Example:
l1 → l2 → l3
returns (l1, l2), (l1, l3), (l2, l3)

### `module_sequential_pairs`  
Returns only **direct neighboring module pairs**.  
Used in adding: [[Sequentail actions]]
  
- Captures **local / adjacent relationships**  
- Skips indirect connections  
  
Example:
l1 → l2 → l3
Returns:
(l1, l2), (l2, l3)
Use this when you care about **immediate layer ordering** (e.g. inserting layers between two modules).

---
 For deleting we have the following actions: 

### `_has_module`
Generic DFS traversal over an FX graph.  
Given a starting node and a direction (via `next_nodes`), it checks whether **any `call_module` node exists** along that path.

### `_has_module_upstream(node)`
Returns `True` if there is a module **before** the given node  
(i.e. reachable via `.all_input_nodes`).


### `_has_module_downstream(node)`
Returns `True` if there is a module **after** the given node  
(i.e. reachable via `.users`).

### `_is_hidden_module(node)`

^7a8eff

Returns `True` if the node is a **hidden layer**, meaning:
- it has a module upstream **and**
- it has a module downstream

In practice, this excludes:
- input layers (no upstream modules)
- output layers (no downstream modules)
