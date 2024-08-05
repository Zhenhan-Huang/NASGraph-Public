import re
import time

import numpy as np
import networkx as nx

from scipy.stats import entropy
from networkx.algorithms import approximation, community, closeness, centrality, distance_measures


class GraphProps(object):
    def __init__(self, G, directed=False, weight=None, dummy=False,
        virtual_source_sink=False, verbose=False):
        self.rgraph = G
        self.verbose = verbose

        if not dummy:
            self.nlayers = self._get_num_layers()
            self.sources, self.sinks = self._get_sourcesink(virtual_source_sink)
            self.weight = weight
            self.nnodes = G.number_of_nodes()
            self.nedges = G.number_of_edges()
            self.degs = self._get_degrees()
            self.mean, self.square_mean = self._get_degree_moments()
            if verbose:
                # for edge in self.rgraph.edges(data=True):
                #     if edge[0] == 'source' or edge[1] == 'sink':
                #         print(edge)
                print(f'Source nodes are {self.sources}')
                print(f'Sink nodes are {self.sinks}')
                print(f'Total number of nodes is {self.nnodes}')
                print(f'Total number of edges is {self.nedges}')

        if not directed:
            self.ftfunc = {
                # num nodes
                'num edges': self.get_num_edges,
                'average degree': self.get_average_degree,
                'cluster coefficient': self.get_cluster,
                'resilience': self.get_resilience,
                'entropy': self.get_degree_entropy,
                'wedge': self.get_wedge_count,
                'gini': self.get_gini_coefficient,
                'density': self.get_density,
                # file name
            }
        else:
            self.ftfunc = {
                # num nodes
                'num edges': self.get_num_edges,
                'average degree': self.get_average_degree,
                'resilience': self.get_resilience,
                'entropy': self.get_degree_entropy,
                'wedge': self.get_wedge_count,
                'gini': self.get_gini_coefficient,
                'density': self.get_density,
                # file name
            }

    def get_feature_names(self):
        return list(self.ftfunc.keys())

    def _get_num_layers(self):
        nodes = self.rgraph.nodes()
        layers = {re.search('(\d+)_', node).group(1) for node in nodes}
        return len(layers)

    def _get_sourcesink(self, virtual_source_sink):
        nodes = self.rgraph.nodes()
        sources, sinks = [], []
        # use 2000 or math.inf
        maxnode,  minnode = 0, 2000
        for node in nodes:
            lid = re.search('(\d+)_', node).group(1)
            if int(lid) < minnode:
                sources = []
                sources.append(node)
                minnode = int(lid)
            elif int(lid) > maxnode:
                sinks = []
                sinks.append(node)
                maxnode = int(lid)
            elif int(lid) == minnode:
                sources.append(node)
            elif int(lid) == maxnode:
                sinks.append(node)

        assert len(sources) != 0, 'Cannot find source nodes'
        assert len(sinks) != 0, 'Cannot find sink nodes'

        if virtual_source_sink:
            new_source_edges = [('source', sr) for sr in sources]
            self.rgraph.add_edges_from(new_source_edges, weight=1./len(new_source_edges))
            new_sink_edges = [(sk, 'sink') for sk in sinks]
            self.rgraph.add_edges_from(new_sink_edges, weight=1./len(new_sink_edges))
            return ['source'], ['sink']
        else:
            return sources, sinks

    def _get_hits(self):
        hubs, authors = nx.hits(self.rgraph)
        return hubs, authors

    def _get_degrees(self):
        degs = self.rgraph.degree(weight=self.weight)
        return np.array([v for v in dict(degs).values()])
        
    def _get_degree_moments(self):
        return self.degs.mean(), np.mean(self.degs ** 2)

    def get_num_edges(self):
        return self.nedges 

    def get_density(self):
        return nx.density(self.rgraph)

    def get_node_connectivity(self):
        vals = []
        for source in self.sources:
            for sink in self.sinks:
                # node connectivity from source to sink
                vals.append(
                    approximation.node_connectivity(self.rgraph, s=source, t=sink))
        return sum(vals) / len(vals)

    def get_degree_assortativity(self):
        return nx.degree_assortativity_coefficient(self.rgraph)

    def get_resistance_distance(self):
        # resistance from source to sink
        return nx.resistance_distance(self.rgraph, 0, self.nnodes-1)

    def get_max_flow(self):
        return nx.maximum_flow(self.rgraph, 0, self.nnodes-1)

    def get_sigma(self):
        # small-world coefficient (sigma) of a given graph
        # NOTE: it is computationally costly
        return nx.sigma(self.rgraph)

    def get_omega(self):
        # small-world coefficient (omega) of a given graph
        # NOTE: it is computationally costly
        return nx.omega(self.rgraph)

    def get_local_constraint(self):
        return nx.local_constraint(self.rgraph, 0, self.nnodes-1)

    def get_avg_constraint(self):
        constraints = nx.constraint(self.rgraph)
        return np.mean(list(constraints.values()))

    def get_avg_closeness_vitality(self):
        cvitalities = nx.closeness_vitality(self.rgraph)
        return np.mean(list(cvitalities.values()))

    def get_closeness_vitality_start(self):
        return nx.closeness_vitality(self.rgraph, node=0)

    def get_closeness_vitality_end(self):
        return nx.closeness_vitality(self.rgraph, node=self.nnodes-1)

    def get_path_source_to_sink(self):
        lens = []
        for source in self.sources:
            for sink in self.sinks:
                len = nx.shortest_path_length(
                    self.rgraph, source=source, target=sink, weight=self.weight)
                lens.append(len)
        return np.mean(lens)

    def get_simple_start_to_end(self):
        ls = []
        for source in self.sources:
            for sink in self.sinks:
                paths = nx.all_simple_paths(self.rgraph, source=source, target=sink)
                for path in map(nx.utils.pairwise, paths):
                    l = 0
                    for pair in path:
                        l += self.rgraph.edges[pair[0], pair[1]]['weight']
                    ls.append(l)
        return np.mean(ls)

    def get_flow_hierarchy(self):
        return nx.flow_hierarchy(self.rgraph)

    def get_max_eigen_laplacian(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            self.lmat = nx.directed_laplacian_matrix(self.rgraph, weight=self.weight)
            self.eigen_spectrum = np.linalg.eig(self.lmat)[0]
        return np.max(self.eigen_spectrum)

    def _compute_path_weight(self, graph, path, weight):
        dist = 0
        for node, nbr in nx.utils.pairwise(path):
            dist_edge = graph[node][nbr][weight] if weight else 1
            dist += dist_edge
        return dist

    def get_edge_disjoint_path_source_to_sink(self):
        lens_all = []
        for source in self.sources:
            for sink in self.sinks:
                paths = nx.edge_disjoint_paths(self.rgraph, source, sink)
                lens = [self._compute_path_weight(self.rgraph, path, weight=self.weight) for path in paths]
                lens_all.append(np.mean(lens))
        return sum(lens_all) / len(lens_all)

    def get_longest_path(self):
        path = nx.dag_longest_path(self.rgraph, weight=self.weight)
        return self._compute_path_weight(self.rgraph, path, weight=self.weight)

    def get_eulerian_path(self):
        lens_all = []
        for source in self.sources:
            dist = 0
            edges = nx.algorithms.euler.eulerian_path(self.rgraph, source)
            for edge in edges:
                dist_edge = edge[self.weight] if self.weight else 1
                dist += dist_edge
            lens_all.append(dist)
        return sum(lens_all) / len(lens_all)

    def get_average_degree(self):
        return self.mean
    
    def get_second_order_moment(self):
        return self.square_mean

    def get_resilience(self):
        return self.square_mean / self.mean

    def get_cluster(self):
        graph_cluster = list(nx.clustering(self.rgraph, weight=self.weight).values())
        return sum(graph_cluster) / len(graph_cluster)

    def get_avg_path(self):
        lengths = np.array([])
        for source, lendict in nx.shortest_path_length(self.rgraph, weight=self.weight):
            sublens = np.array(list(lendict.values()))
            # remove self-loop paths and unreacheable path
            sublens = sublens[sublens > 0]
            lengths = np.concatenate((lengths, sublens))
        return np.mean(lengths)

    def get_heterogeneity(self):
        return (self.square_mean - self.mean ** 2) / self.mean
    
    def get_bmodule(self):
        bipartition = community.kernighan_lin_bisection(
            self.rgraph, partition=None, max_iter=100, seed=1        
        )
        bimodule = community.quality.modularity(
            self.rgraph, bipartition        
        )
        return bimodule

    def get_gmodule(self):
        greedy_partition = community.modularity_max.greedy_modularity_communities(self.rgraph)
        gmodule = community.quality.modularity(self.rgraph, greedy_partition)
        return gmodule

    def get_degree_entropy(self):
        ps = self.degs / self.nedges
        return entropy(ps) / self.nnodes

    def get_wedge_count(self):
        prod = self.degs * (self.degs - 1) / 2
        return sum(prod)
    
    def get_power_law_exponent(self):
        mink = self.degs.min()
        alpha = 1 + self.nnodes / sum(np.log(self.degs / mink))
        return alpha

    def get_gini_coefficient(self):
        sorted_degs = sorted(self.degs)
        gini = sum([(i+1) * sorted_degs[i] for i in range(self.nnodes)]) / (self.nedges * self.nnodes) \
            - (self.nnodes + 1) / self.nnodes
        return gini

    def get_average_node_connectivity(self):
        return nx.average_node_connectivity(self.rgraph)

    def node_connectivity_start_end(self):
        return approximation.local_node_connectivity(self.rgraph, 0, self.nnodes-1)

    def get_average_edge_connectivity(self):
        return nx.algorithms.connectivity.connectivity.edge_connectivity(self.rgraph)
    
    def get_average_closeness_centrality(self):
        self.ccs = closeness.closeness_centrality(
            self.rgraph, wf_improved=False, distance=self.weight)
        return sum(self.ccs.values()) / len(self.ccs)

    def get_closeness_centrality_source(self):
        vals = []
        for source in self.sources:
            vals.append(self.ccs[source])
        return sum(vals) / len(vals)

    def get_closeness_centrality_sink(self):
        vals = []
        for sink in self.sinks:
            vals.append(self.ccs[sink])
        return sum(vals) / len(vals)

    def get_average_closeness_centrality_wf(self):
        cs = closeness.closeness_centrality(self.rgraph, wf_improved=True).values()
        return np.mean(list(cs))

    def get_average_eccentricity(self):
        varepsilons = distance_measures.eccentricity(self.rgraph).values()
        return np.mean(list(varepsilons))

    def get_diameter(self):
        return distance_measures.diameter(self.rgraph)

    def get_radius(self):
        return distance_measures.radius(self.rgraph)

    def get_average_degree_centrality(self):
        return np.mean(list(nx.degree_centrality(self.rgraph).values()))

    def get_start_degree_centrality(self):
        odcs = nx.out_degree_centrality(self.rgraph)
        ret = 0
        for source in self.sources:
            ret += odcs[source]
        return ret

    def get_end_degree_centrality(self):
        idcs = nx.in_degree_centrality(self.rgraph)
        ret = 0
        for sink in self.sinks:
            ret += idcs[sink]
        return ret

    def get_average_edge_betweenness_centrality(self):
        ecs = centrality.edge_betweenness_centrality(
            self.rgraph, weight=self.weight)
        return sum(ecs.values()) / len(ecs)


    def get_average_node_betweenness_centrality(self):
        self.bcs = centrality.betweenness_centrality(self.rgraph, weight=self.weight)
        return sum(self.bcs.values()) / len(self.bcs)

    def get_node_betweenness_centrality_source(self):
        nbs = [self.bcs[source] for source in self.sources]
        return sum(nbs) / len(nbs)

    def get_node_betweenness_centrality_sink(self):
        nbs = [self.bcs[sink] for sink in self.sinks]
        return sum(nbs) / len(nbs)  

    def get_central_point_of_dominance(self):
        bcs = np.array(list(centrality.betweenness_centrality(self.rgraph, weight=self.weight).values()))
        cpd = np.sum(bcs.max() - bcs) / (self.nnodes - 1)
        return cpd

    def get_average_core_number(self):
        core_nums = list(nx.core_number(self.rgraph).values())
        return np.mean(core_nums)
    
    def get_laplacian_min_spectrum(self):
        laplacian_eigenvalues = nx.linalg.spectrum.laplacian_spectrum(self.rgraph)
        return laplacian_eigenvalues.min()
    
    def get_laplacian_max_spectrum(self):
        laplacian_eigenvalues = nx.linalg.spectrum.laplacian_spectrum(self.rgraph)
        return laplacian_eigenvalues.max()
    
    def get_transitivity(self):
        return nx.algorithms.cluster.transitivity(self.rgraph)
    
    def get_local_efficiency(self):
        efficiency_list = (self._cal_global_efficiency(self.rgraph.subgraph(self.rgraph[v])) for v in self.rgraph)
        return sum(efficiency_list) / self.nnodes   

    def get_global_efficiency(self):
        return self._cal_global_efficiency(self.rgraph)

    def get_pagerank_start(self):
        self.pr = nx.pagerank(self.rgraph, weight=self.weight)
        prs_sr = []
        for source in self.sources:
            prs_sr.append(self.pr[source])
        return np.mean(prs_sr)

    def get_pagerank_end(self):
        prs_si = []
        for sink in self.sinks:
            prs_si.append(self.pr[sink])
        return np.mean(prs_si)

    def _cal_global_efficiency(self, graph):
        n = len(graph)
        denom = n * (n - 1)
        if denom != 0:
            lengths = nx.all_pairs_shortest_path_length(graph)
            g_eff = 0
            for source, targets in lengths:
                for target, distance in targets.items():
                    if distance > 0:
                        g_eff += 1 / distance
            g_eff /= denom
        else:
            g_eff = 0
        return g_eff

    def get_num_triangles(self):
        return np.sum(nx.triangles(self.rgraph).values())

    def get_all_features(self):
        features = [self.nnodes]
        for name, func in self.ftfunc.items():
            time_st = time.time()
            features.append(func())
            time_ed = time.time()
            if self.verbose:
                print(f'Computing {name} takes {(time_ed-time_st)/60:.2f} mins')
                print(f'{name} = {features[-1]:.6f}')
        return features
        
    def get_feature_values(self, selected_idxs):
        features = []
        for i in selected_idxs:
            feature = list(self.ftfunc.values())[i]()
            features.append(feature)
        return features
    
    def get_feature_names(self):
        names = [key for key in self.ftfunc]
        names = ['num nodes'] + names
        return names

    # average number is 1/N, N is the number of nodes
    def get_hubs_source(self):
        hubs, _ = nx.hits(self.rgraph)
        return hubs[0]

    # average number is 1/N, N is the number of nodes
    def get_authors_sink(self):
        _, authorities = nx.hits(self.rgraph)
        return authorities[self.nnodes-1]
