# Assumptions
# - The nodes are topologically sorted while being passed as input, i.e., parents of a node should have a lower idx than the child node
# - Uniform decay for all genes
# - Steady state calculation - SERGIO
# -

import os
import numpy as np
import pandas as pd
import networkx as nx
import random


class dynSim:
    def __init__(self):
        ### 2 MRs and 2 Genes
        self.g = nx.DiGraph()
        self.g.add_node(0, b=0.20165, t='mr')
        self.g.add_node(1, b=0.25316, t='mr')
        self.g.add_node(2, b=np.random.uniform(0.1, 1.0), t='g', ki=[1.73104, 2.64137], p_gm=3, p_kt=10)
        self.g.add_node(3, b=np.random.uniform(0.1, 1.0), t='g', ki=[2.34309], p_gm=2, p_kt=12)
        self.g.add_edges_from([[0, 2], [1, 2], [2, 3]])
    
        self.decay = 0.8
        self.calc_steady_states()
        
    def calc_steady_states(self):
        
        for i in self.g.nodes(data=True):
            if i[1]['t'] == 'mr':
                self.g.nodes[i[0]]['e_x'] = i[1]['b'] / self.decay
                self.g.nodes[i[0]]['c'] = i[1]['b'] / self.decay
            else:
                node = self.g.nodes[i[0]]
                node['e_x'] = self.calc_pij(i[0]) / self.decay
                node['c'] = node['e_x']
                node['p_kd'] = np.math.log(2) / node['p_gm']
                node['p_c'] = node['p_kt'] * node['c'] / node['p_kd']

    def calc_pij(self, idx, _hill=1):
        node = self.g.nodes()[idx]
        if node['t'] =='mr':
            return node['b']
        regs = sorted(self.g.predecessors(idx))
        s_pij = 0
        for i in range(len(regs)):
            node_j = self.g.nodes()[regs[i]]
            s_pij += (node['ki'][i] * (node_j['c'] ** _hill)) /(node_j['c'] ** _hill + node_j['c'] ** _hill)
        # p_i = s_pij + g.nodes()[idx]['b']
        return s_pij

    def calc_x_t(self, idx, delta=0.1):
        node = self.g.nodes()[idx]
        p_i = self.calc_pij(idx) + node['b']
        node['c']= node['c'] + (p_i - self.decay*node['c']) * delta
        if node['t'] == 'g':
            node['p_c'] = node['p_c'] + node['p_kt']*node['c'] - node['p_kd']*node['p_c']

    def ret_vals(self):
        vals = []
        prot_vals = []
        for i in self.g.nodes():
            vals.append(self.g.nodes()[i]['c'])
            if self.g.nodes()[i]['t'] == 'g':
                prot_vals.append(self.g.nodes()[i]['p_c'])
        return np.array(vals), np.array(prot_vals)