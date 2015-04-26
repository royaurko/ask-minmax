'''Write the problems in json format'''
import json


def write(f):
    problems = dict()
    problems['matching'] = {'bipartite':{}, 'non-bipartite':{}, 'b-matching':{}, 'stable matching':{}}
    problems['matching']['k-way matching'] = {}
    problems['matching']['Hypergraph matching'] = {}
    problems['flows'] = {'maximum flow':{}}
    problems['routing'] = {'packet routing':{}, 'dial-a-ride':{}}
    problems['routing']['routing with congestion'] = {}
    problems['routing']['routing with conflict'] = {}
    problems['cuts'] = {'maximum cut':{}, 'minimum cut':{}, 'sparsest cut':{}}
    problems['tsp'] = {'with repetition':{}, 'without repetition':{}}
    problems['tsp']['assymetric'] = {}
    problems['tsp']['symmetric'] = {}
    problems['tsp']['metric'] = {}
    problems['packing and covering'] = {'set cover':{}, 'facility location':{}}
    problems['packing and covering']['knapsack'] = {}
    problems['packing and covering']['bin packing'] = {'1D':{}, '2D':{}, '3D':{}, 'strip packing':{}}
    problems['k-server'] = {'paging':{}, 'weighted':{}, 'euclidean':{}, 'offline':{}}
    problems['k-server']['online with d-lookahead'] = {}
    problems['coloring'] = {'planar graphs':{}, 'bounded tree width':{}, 'perfect graph':{}}
    problems['scheduling'] = {'preemptive':{}, 'non-preemptive':{}, 'single machine':{}}
    problems['scheduling']['multiple machines'] = {}
    problems['scheduling']['with start time'] = {}
    json.dump(problems, f)



if __name__ == '__main__':
    fname = 'problems.json'
    f = open(fname, 'w')
    write(f)
