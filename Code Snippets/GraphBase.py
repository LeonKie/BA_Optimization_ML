import numpy as np
import itertools

# igraph
import igraph


class GraphBase(object):
    """
    Created by: Tim Stahl
    Created on: 28.09.2018

    Documentation: (Short) Database for writing, storing as well as retrieving nodes and edges.

                   (Long) This class serves the following main purposes:
                    * interfacing the graph-tool class (providing fast c++ operations) - all the contact with this api
                      should be managed within this class
                    * providing simple methods to store and access nodes as well as edges (together with their
                      trajectory specific information / properties)
                    * the class should host all information required for the online execution (storing and loading this
                      class), therefore relevant environment information are stored as class members (ref. line, ...)
                    * online methods, e.g. graph search and filtering (obstacles), can be triggered with class member
                      functions

    """

    def __init__(self,
                 lon_layers: int,
                 lat_offset: float,
                 num_layers: int,
                 refline: np.array,
                 track_width_right: np.array,
                 track_width_left: np.array,
                 lat_resolution: float,
                 normvec_normalized: np.array,
                 alpha_mincurv: np.array,
                 s_raceline: np.array,
                 sampled_resolution: float,
                 vel_raceline: np.array,
                 vel_decrease_lat: float,
                 veh_width: float,
                 veh_length: float,
                 veh_turn: float,
                 md5_params: str,
                 graph_id: str,
                 glob_rl_clsd: np.ndarray,
                 virt_goal_node=True,
                 virt_goal_node_cost=200.0,
                 min_plan_horizon: float = 200.0,
                 plan_horizon_mode: str = 'distance'):
        """ initialize graph base
        """
        # Version number (Increases, when new options are added / changed)
        self.VERSION = 0.2

        # general (public) parameters
        self.lon_layers = lon_layers
        self.lat_offset = lat_offset
        self.num_layers = num_layers
        self.refline = refline
        self.track_width_right = track_width_right
        self.track_width_left = track_width_left
        self.lat_resolution = lat_resolution
        self.normvec_normalized = normvec_normalized
        self.alpha_mincurv = alpha_mincurv
        self.s_raceline = s_raceline
        self.sampled_resolution = sampled_resolution
        self.vel_raceline = vel_raceline
        self.vel_decrease_lat = vel_decrease_lat
        self.veh_width = veh_width
        self.veh_length = veh_length
        self.veh_turn = veh_turn
        self.raceline_index = None
        self.md5_params = md5_params
        self.graph_id = graph_id
        self.glob_rl_clsd = glob_rl_clsd
        self.virt_goal_node = virt_goal_node
        self.virt_goal_node_cost = virt_goal_node_cost
        self.min_plan_horizon = min_plan_horizon
        self.plan_horizon_mode = plan_horizon_mode

        # calculate raceline
        self.raceline = refline + normvec_normalized * alpha_mincurv[:, np.newaxis]

        # initialize graph-tool object
        self.__g = igraph.Graph()
        self.__g.to_directed()

        # copy of original graph (for filtering)
        self.__g_orig = None

        # set of filter graphs
        self.__g_filter = dict()

        # dictionary holding all nodes and corresponding information
        self.__virtual_layer_node = dict()

        # number of nodes per layer
        self.nodes_in_layer = dict()

    # ------------------------------------------------------------------------------------------------------------------
    # BASIC NODE FUNCTIONS ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def add_node(self,
                 layer: int,
                 node_number: int,
                 pos: np.array,
                 psi: float,
                 raceline_index=None):
        """ stores a node which will later hold all relevant information
        """

        # if plain (non-filtered) graph is not active, switch to it
        if self.__g_orig is not None:
            self.__g = self.__g_orig

        # create vertex in graph-tool object and store related information / properties
        self.__g.add_vertex(name=str((layer, node_number)),
                            position=pos,
                            psi=psi,
                            raceline=(raceline_index == node_number),
                            node_id=node_number,
                            layer_id=layer)

        # update the max node index for the given layer (if the value is larger than a possibly existing stored one)
        self.nodes_in_layer[layer] = max(self.nodes_in_layer.get(layer, 0), node_number + 1)

        # if virtual layer node activated
        if self.virt_goal_node:
            # check if virtual layer node exists
            if layer not in self.__virtual_layer_node.keys():
                # add virtual goal node for given layer
                vv = "v_l" + str(layer)
                self.__g.add_vertex(name=vv,
                                    layer_id=layer)

                # store reference in dict
                self.__virtual_layer_node[layer] = vv
            else:
                vv = self.__virtual_layer_node[layer]

            # add edge between virtual and generated node
            offline_cost = abs(raceline_index - node_number) * self.lat_resolution * self.virt_goal_node_cost

            self.__g.add_edge(source=str((layer, node_number)),
                              target=vv,
                              virtual=1,
                              start_layer=layer,
                              offline_cost=offline_cost)

    def add_layer(self,
                  layer: int,
                  pos_multi: np.array,
                  psi: np.array,
                  raceline_index: int):
        """ add several nodes belonging to one layer (several positions provided via "pos_multi". no children assumed.
        """
        for i in range(len(pos_multi)):
            self.add_node(layer=layer,
                          node_number=i,
                          pos=pos_multi[i],
                          psi=psi[i],
                          raceline_index=raceline_index)

        # update the max node index for the given layer (if the value is larger than a possibly existing stored one)
        self.nodes_in_layer[layer] = max(self.nodes_in_layer.get(layer, 0), len(pos_multi))

    def get_node_info(self,
                      layer: int,
                      node_number: int,
                      return_child=False,
                      return_parent=False,
                      active_filter: str = "current") -> tuple:
        """ return information stored for a specific node
        """

        if active_filter is None:
            g = self.__g_orig
        elif active_filter == "current":
            g = self.__g
        else:
            g = self.__g_filter[active_filter]

        # Check for invalid ID
        try:
            node = g.vs.find(str((layer, node_number)))
        except ValueError as e:
            print("KeyError - Could not find requested node ID! " + str(e))
            return None, None, None, None, None

        pos = node['position']
        psi = node['psi']
        raceline = node['raceline']

        # retrieve nodes children
        if return_child:
            idx_children = g.successors(node.index)
            children = [(g.vs[v]["layer_id"], g.vs[v]["node_id"]) for v in idx_children
                        if g.vs[v]["node_id"] is not None]
        else:
            children = None

        # retrieve nodes parents
        if return_parent:
            idx_parents = g.predecessors(node.index)
            parents = [(g.vs[v]["layer_id"], g.vs[v]["node_id"]) for v in idx_parents
                       if g.vs[v]["node_id"] is not None]
        else:
            parents = None

        return pos, psi, raceline, children, parents

    def get_layer_info(self,
                       layer: int):
        """ return information for all nodes in the specified layer
        """
        pos_list = []
        psi_list = []
        raceline_list = []
        children_list = []

        for i in range(self.nodes_in_layer[layer]):
            pos, psi, raceline, children, _ = self.get_node_info(layer=layer,
                                                                 node_number=i,
                                                                 return_child=True,
                                                                 active_filter=None)

            if pos is not None:
                pos_list.append(pos)
                psi_list.append(psi)
                raceline_list.append(raceline)
                children_list.append(children)

        return pos_list, psi_list, raceline_list, children_list

    def get_closest_nodes(self,
                          pos: np.array,
                          limit=1,
                          fixed_amount=True):
        """ search for the n closest points in the graph for a given coordinate (x, y), with two modes as follows:
                - fixed_amount=True:  return the n closest points (amount specified by "limit")
                - fixed_amount=False: return _all_ points within a radius specified by "limit" in meters
        """
        # extract coordinates and calculate squared distances
        node_indexes, node_positions = \
            zip(*[(node_idx, node_pos) for node_idx, node_pos in enumerate(self.__g.vs['position'])
                  if node_pos is not None])
        node_positions = np.vstack(node_positions)
        distances2 = np.power(node_positions[:, 0]-pos[0], 2) + np.power(node_positions[:, 1]-pos[1], 2)

        if fixed_amount:
            # find k indexes holding minimum squared distances
            idx = np.argpartition(distances2, limit)[:limit]
        else:
            # get all elements within radius "limit"
            ref_radius = limit*limit
            idx = np.argwhere(distances2 < ref_radius).reshape(1, -1)[0]

        # get actual distances of those k nodes
        distances = np.sqrt(distances2[idx])

        # return matching node ids (layer and node number)
        nodes = [(self.__g.vs[node_indexes[v]]["layer_id"], self.__g.vs[node_indexes[v]]["node_id"]) for v in idx
                 if self.__g.vs[node_indexes[v]]["node_id"] is not None]
        return nodes, distances

    def get_nodes(self) -> list:
        """ Returns a list of all stored nodes
        """
        return [(v["layer_id"], v["node_id"]) for v in self.__g.vs if v["node_id"] is not None]

    # ------------------------------------------------------------------------------------------------------------------
    # BASIC EDGE FUNCTIONS ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def add_edge(self,
                 start_layer: int,
                 start_node: int,
                 end_layer: int,
                 end_node: int,
                 **kwargs):  # Optionally pass argument "spline_coeff", "spline_coord", "offline_cost"
        """ stores an edge with given identifier
        """

        # if plain (non-filtered) graph is not active, switch to it
        if self.__g_orig is not None:
            self.__g = self.__g_orig

        sn = str((start_layer, start_node))
        en = str((end_layer, end_node))

        # create edge, if not existent yet
        edge_id = self.__g.get_eid(sn, en, error=False)
        if edge_id == -1:
            self.__g.add_edge(start_layer=start_layer,
                              virtual=0,
                              source=sn,
                              target=en,
                              spline_coeff=None,
                              spline_length=None,
                              spline_param=None,
                              offline_cost=None)
            edge_id = self.__g.get_eid(sn, en, error=False)

        # check for optional arguments
        if 'spline_coeff' in kwargs:
            self.__g.es[edge_id]['spline_coeff'] = kwargs.get('spline_coeff')

        if 'spline_x_y_psi_kappa' in kwargs:
            # calculate length
            el_lengths = np.sqrt(np.sum(np.power(
                np.diff(kwargs.get('spline_x_y_psi_kappa')[:, 0:2], axis=0), 2), axis=1))

            # store total spline length
            self.__g.es[edge_id]['spline_length'] = np.sum(el_lengths)

            #  append zero to el length array, in order to reach equal array length
            el_lengths = np.append(el_lengths, 0)

            # generate proper formatted numpy array containing spline data
            self.__g.es[edge_id]['spline_param'] = \
                np.column_stack((kwargs.get('spline_x_y_psi_kappa'), el_lengths))

        if 'offline_cost' in kwargs:
            self.__g.es[edge_id]['offline_cost'] = kwargs.get('offline_cost')

    # "update_edge" equals the function "add_edge"
    update_edge = add_edge

    def get_edge(self,
                 start_layer: int,
                 start_node: int,
                 end_layer: int,
                 end_node: int) -> tuple:
        """ Retrieve the information stored for a specified edge
        """
        sn = str((start_layer, start_node))
        en = str((end_layer, end_node))

        edge_id = self.__g.get_eid(sn, en)  # if we want to catch errors, add "error=False" and check for returned "-1"
        edge = self.__g.es(edge_id)

        spline_coeff = edge['spline_coeff'][0]
        spline_param = edge['spline_param'][0]
        offline_cost = edge['offline_cost'][0]
        spline_length = edge['spline_length'][0]

        return spline_coeff, spline_param, offline_cost, spline_length

    def factor_edge_cost(self,
                         start_layer: int,
                         start_node: int,
                         end_layer: int,
                         end_node: int,
                         cost_factor: float,
                         active_filter: str = "current"):
        """ Update the cost of a specified edge
        """

        if active_filter is None:
            g = self.__g_orig
        elif active_filter == "current":
            g = self.__g
        else:
            g = self.__g_filter[active_filter]

        sn = str((start_layer, start_node))
        en = str((end_layer, end_node))

        try:
            edge_id = g.get_eid(sn, en)
            g.es[edge_id]['offline_cost'] *= cost_factor
        except ValueError:
            # catch errors, when nodes are not present anymore (e.g. filtering)
            pass

        return

    def remove_edge(self,
                    start_layer: int,
                    start_node: int,
                    end_layer: int,
                    end_node: int):
        """ Remove the specified edge from the graph (handle with caution - only for the offline part)
        """

        sn = str((start_layer, start_node))
        en = str((end_layer, end_node))

        edge_id = self.__g.get_eid(sn, en)
        self.__g.delete_edges(edge_id)

        return

    def get_intersec_edges_in_range(self,
                                    start_layer: int,
                                    end_layer: int,
                                    obstacle_pos: np.array,
                                    obstacle_radius: float,
                                    remove_filters: bool = True):
        """ Determine all edges between start_layer and end_layer (+/- lon_layers), that intersect with the obstacle
            specified by a pos and a radius

            IMPORTANT: Edge ids are just valid in the same graph --> when filtering make sure to apply filter before
                       intersecting edge detection and directly process edge ids afterwards
        """

        # determine start and end layer based on longitudinal steps
        cor_start_layer = start_layer - self.lon_layers + 1
        if cor_start_layer < 0:
            cor_start_layer += self.num_layers
        cor_end_layer = end_layer + self.lon_layers - 1
        if cor_end_layer > self.num_layers:
            cor_end_layer -= self.num_layers

        if remove_filters:
            g = self.__g_orig
        else:
            g = self.__g

        # # select subset of edges to check for collision
        # if start_layer < end_layer:
        #     selected_edges = g.es.select(virtual_eq=0, start_layer_ge=start_layer, start_layer_le=end_layer-1)
        # else:
        #     # assuming start line overlaps (next lap)
        #     layer_set = list(range(start_layer, self.num_layers)) + list(range(0, end_layer+1))
        #     selected_edges = g.es.select(virtual_eq=0, start_layer_in=layer_set)
        if start_layer < end_layer:
            selected_nodes = g.vs.select(layer_id_ge=start_layer, layer_id_le=end_layer)
        else:
            # assuming start line overlaps (next lap)
            layer_set = list(range(start_layer, self.num_layers)) + list(range(0, end_layer + 1))
            selected_nodes = g.vs.select(layer_id_in=layer_set)

        sub_g = g.induced_subgraph(selected_nodes)
        selected_edges = sub_g.es

        # obstacle reference is sum of radius of both objects plus an offset to cope with discretization
        # NOTE: formula given by isosceles triangle -> a = sqrt(h^2 + c^2/4)
        obstacle_ref = np.power(obstacle_radius + self.veh_width / 2, 2) + np.power(self.sampled_resolution, 2)/4
        intersec_edges = []
        # for all edges (with applied filter)
        for edge in selected_edges:
            # check edge for collision
            param = edge["spline_param"]

            # check if edge is not a virtual one
            if param is not None:
                param = param
                x = param[:, 0] - obstacle_pos[0]
                y = param[:, 1] - obstacle_pos[1]
                distances2 = x * x + y * y
                if any(distances2 <= obstacle_ref):
                    intersec_edges.append([sub_g.vs[edge.source]["name"], sub_g.vs[edge.target]["name"]])

        return intersec_edges

    def get_edges(self) -> list:
        """ Returns a list of all stored edges
        """
        return [(self.__g.vs[edge.source]["layer_id"], self.__g.vs[edge.source]["node_id"],
                 self.__g.vs[edge.target]["layer_id"], self.__g.vs[edge.target]["node_id"]) for edge in self.__g.es
                if self.__g.vs[edge.source]["node_id"] is not None and self.__g.vs[edge.target]["node_id"] is not None]

    # ------------------------------------------------------------------------------------------------------------------
    # FILTERING --------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def init_filtering(self):
        """ stores a version of the graph as an original copy
        """
        self.__g_orig = self.__g.copy()

    def set_node_filter_layers(self,
                               start_layer: int,
                               end_layer: int,
                               applied_filter="default",
                               base: str = None):
        """ Initializes a filter with the nodes belonging to layers in range "start_layer" to "end_layer" to visible.
            If no "applied_filter" is provided, the internal active node filter is used.

            Input:
            - start_layer:      index of the first layer to be included in the filtered graph
            - end_layer:        index of the last layer to be included in the filtered graph
            - applied_filter:   name of the filtered set
            - base:             provide a string of a filtered set here, to build the new filtered graph upon an
                                existing one (if not set, build from scratch)
        """
        if self.__g_orig is None:
            self.__g_orig = self.__g.copy()

        if base is None:
            g = self.__g_orig
        elif base == "current":
            g = self.__g
        else:
            if base in self.__g_filter.keys():
                g = self.__g_filter[base]
            else:
                g = self.__g_orig

        if start_layer < end_layer:
            selected_nodes = g.vs.select(layer_id_ge=start_layer, layer_id_le=end_layer)
        else:
            # assuming start line overlaps (next lap)
            layer_set = list(range(start_layer, self.num_layers)) + list(range(0, end_layer + 1))
            selected_nodes = g.vs.select(layer_id_in=layer_set)

        self.__g_filter[applied_filter] = g.induced_subgraph(selected_nodes)

    def remove_nodes_filter(self,
                            layer_ids: list,
                            node_ids: list,
                            applied_filter="default",
                            base: str = None):
        """ update the nodes specified by the lists "layer_ids" and "node_ids" in the filter "applied_filter",
            listed nodes are removed
        """
        if self.__g_orig is None:
            self.__g_orig = self.__g.copy()

        if base is None:
            g = self.__g_orig
        elif base == "current":
            g = self.__g
        else:
            if base in self.__g_filter.keys():
                g = self.__g_filter[base]
            else:
                g = self.__g_orig

        name_ref = [str((x, y)) for x, y in zip(layer_ids, node_ids)]

        selected_nodes = g.vs.select(name_notin=name_ref)
        self.__g_filter[applied_filter] = g.induced_subgraph(selected_nodes)

    def init_edge_filter(self,
                         disabled_edges,
                         applied_filter="default",
                         base: str = None):
        """ Initializes a filter with the edges in the list "disabled_edges" being removed

            Input:
            - disabled_edges:   list of edges (start and end node) to be not present in the resulting filtered graph
            - applied_filter:   name of the filtered set
            - base:             provide a string of a filtered set here, to build the new filtered graph upon an
                                existing one (if not set, build from scratch)
        """

        if self.__g_orig is None:
            self.__g_orig = self.__g.copy()

        if base is None:
            g = self.__g_orig
        elif base == "current":
            g = self.__g
        else:
            g = self.__g_filter[base]

        # Determine edge ids to be deleted
        edge_ids = g.get_eids(pairs=disabled_edges)

        self.__g_filter[applied_filter] = g.copy()
        self.__g_filter[applied_filter].delete_edges(edge_ids)

    def deactivate_filter(self):
        """ deactivate any active node filter
        """
        self.__g = self.__g_orig

    def activate_filter(self,
                        applied_filter="default"):
        """ apply a certain filter
        """
        if self.__g_orig is None:
            self.__g_orig = self.__g.copy()

        self.__g = self.__g_filter[applied_filter]

    # ------------------------------------------------------------------------------------------------------------------
    # GRAPH SEARCH -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def search_graph(self,
                     start_node: int,
                     end_node: int,
                     max_solutions=1):
        """ search a cost minimal path in the graph (returns "None"s if graph search problem is infeasible)

            Input:
            start_node:         node id (internal graph node type identifier)
            end_node:           node id (internal graph node type identifier)
            max_solutions:      maximum number of path solutions to be returned (NOTE: 1 is the fastest!)
            max_cost_diff:      amount of cost difference over the total paths to be considered for multiple returned
                                paths (NOTE: the larger this value, the slower the search!)
        """

        v_list_iterator = self.__g.get_shortest_paths(start_node,
                                                      to=end_node,
                                                      weights="offline_cost",
                                                      output="vpath")
        # NOTE: self.__g.get_eid(path=v_list_iterator) -> returns edge ids

        # Check if the graph search problem is feasible
        if not v_list_iterator or not any(v_list_iterator[0]):
            positions_list = None
            node_ids_list = None
        else:
            positions_list = []
            node_ids_list = []
            for v_list in itertools.islice(v_list_iterator, max_solutions):
                # we want to extract the nodes IDs and their positions
                node_ids = []
                pos = []
                # loop through all edges in sublist
                for node_id in v_list:
                    # if node is not a virtual one
                    node = self.__g.vs[node_id]
                    if node["node_id"] is not None:
                        node_ids.append([node["layer_id"], node["node_id"]])
                        pos.append(node["position"])

                # append this path to list
                positions_list.append(pos)
                node_ids_list.append(node_ids)

        return positions_list, node_ids_list

    def search_graph_layer(self,
                           start_layer: int,
                           start_node: int,
                           end_layer: int,
                           max_solutions=1):
        """ interfaces the graph search function and handles the problem of finding the goal node in the specified end
            layer. Either sequentially testing the nodes in the goal layer or using the virtual goal layer (if active)
        """
        positions_list = None
        node_ids_list = None

        # Try to extract start node, if blocked return as unsolvable
        try:
            start_node_id = self.__g.vs.find(str((start_layer, start_node)))
        except ValueError:
            return None, None

        if self.virt_goal_node:
            # Determine virtual goal layer node (Note: the virtual node is automatically removed after graph search)
            virt_node = self.__virtual_layer_node[end_layer]

            # trigger graph search to virtual goal node
            positions_list, node_ids_list = self.search_graph(start_node=start_node_id,
                                                              end_node=virt_node,
                                                              max_solutions=max_solutions)

        else:
            # Search in defined goal layer, if search does not result in a solution, check pts nxt to raceline
            # Search through nodes in end_layer until a solution is found or no node is left (computationally expensive)
            end_node = None
            while True:
                if end_node is None:
                    end_node = self.raceline_index[end_layer]
                elif end_node <= self.raceline_index[end_layer]:
                    # Race line point seems to be blocked -> look through points with smaller index
                    if end_node <= 0:
                        end_node = self.raceline_index[end_layer] + 1
                    else:
                        end_node -= 1
                else:
                    # Race line point & points with smaller index are blocked -> look through points with larger index
                    end_node += 1

                    # Check if node exists
                    if self.get_node_info(layer=end_layer, node_number=end_node)[0] is None:
                        break

                end_node_id = self.__g.vs.find(str((end_layer, end_node)))

                # Trigger graph search (if goal node exists)
                if self.get_node_info(layer=end_layer, node_number=end_node)[0] is not None:
                    positions_list, node_ids_list = self.search_graph(start_node=start_node_id,
                                                                      end_node=end_node_id,
                                                                      max_solutions=max_solutions)

                # If solution is found, return (otherwise search for goal state in next layer)
                if node_ids_list is not None:
                    break

        return positions_list, node_ids_list


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
