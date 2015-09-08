#!/usr/bin/env python

### IMPORTS ###
import sys, os
from functools import partial

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..'))
from peas.methods.neat import NEATPopulation, NEATGenotype
# from peas.methods.neatpythonwrapper import NEATPythonPopulation
from peas.tasks.polebalance import PoleBalanceTask


class DJTempoMatch(object):

    def __init__(self):

        # Create a Topology
        inputs = 256
        hiddens = 100
        outputs = 5
        bias    = 1
        topology = self._get_topology(inputs=inputs, hiddens=hiddens, outputs=outputs, bias=bias)

        # Create a factory for genotypes (i.e. a function that returns a new
        # instance each time it is called)
        genotype = lambda: NEATGenotype(inputs=inputs,
                                outputs=outputs,
                                feedforward=False,
                                weight_range=(-1., 1.),
                                types=['tanh'],
                                topology = topology,
                                bias_as_node = bias)

        # Create a population
        self.pop = NEATPopulation(genotype, popsize=10)

        # Create a task
        self.dpnv = PoleBalanceTask(velocities=True,
                               max_steps=10,
                               penalize_oscillation=True)

    def _get_topology(self, inputs=5, hiddens = 3, outputs=2, bias=0):
        topology = []

        for i in range(inputs + bias):
            for j in range(inputs + bias, inputs + bias + hiddens):
                topology.append([i, j])

        # Recurrent Network
        for i in range(inputs + bias, inputs + bias + hiddens):
            for j in range(inputs + bias + hiddens, inputs + bias + hiddens * 2):
                    topology.append([i, j])
                    topology.append([j, i])

        for i in range(inputs + bias, inputs + bias + hiddens):
            for j in range(inputs + bias + hiddens * 2, inputs + bias + hiddens * 2 + outputs):
                topology.append([i, j])

        return topology

    def start(self, gen=100):
        # Run the evolution, tell it to use the task as an evaluator
        self.pop.epoch(generations=gen, evaluator=self.dpnv, solution=self.dpnv, callback=self.__new_generation)

    def __new_generation(self, pop):
        print pop.champions[-1]
        pop.champions[-1].visualize("img.png");

if __name__ == "__main__":
    dj = DJTempoMatch()
    dj.start(10)
