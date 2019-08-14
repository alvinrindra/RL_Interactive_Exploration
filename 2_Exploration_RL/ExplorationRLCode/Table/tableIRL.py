# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:43:35 2015

@author: cruz

"""
#Libraries Declaration
import numpy as np
import matplotlib.pyplot as plt

from classes.Variables import *
from classes.Scenario import Scenario
from classes.Agent import Agent
from classes.DataFiles import DataFiles
from itertools import cycle

resultsFolder = 'results/'
files = DataFiles()

def plotRewards(flex_param, param_name, learning_alg, explorationstrat):

    if explorationstrat == explorationstrat_epsgreedy:
        explorationstrat_name = "epsilon-greedy"
    elif explorationstrat == explorationstrat_softmax:
        explorationstrat_name = "Softmax"
    elif explorationstrat == explorationstrat_vdbe:
        explorationstrat_name = "VDBE"
    elif explorationstrat == explorationstrat_vdbe_softmax:
        explorationstrat_name = "VDBE-Softmax"
    else:
        test = 1   #todo error

    if learning_alg == learning_alg_qlearning:
        learning_alg_name = "Q-Learning"
    elif learning_alg == learning_alg_sarsa:
        learning_alg_name = "Sarsa"
    else:
        test = 1 #todo error



    plotname = learning_alg_name + " - " + explorationstrat_name
    plt.figure(plotname)
    plt.suptitle(plotname)

    plt.rcParams['font.size'] = 38
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)

    lines = ["-","--","-.",":"]
    colors = ['k','y','m','c','b','g','r']
    markers = ['x','8','v','^','*','s','o']
    markershift = [0, 20, 40, 60, 80]
    linecycler = cycle(lines)
    colorcycler = cycle(colors)
    markercycle = cycle(markers)
    markershiftcycle = cycle(markershift)


    for x in flex_param:
        dataRL = np.genfromtxt(resultsFolder + 'rewardsRL_' + str(x) + '.csv', delimiter=',')
        meansRL = np.mean(dataRL, axis=0)
        convolveSet = 20
        convolveRL = np.convolve(meansRL, np.ones(convolveSet)/convolveSet)
        # plt.plot(meansRL, label = 'Average reward RL, ' + str(param_name) + ' = ' + str(x), linestyle = next(linecycler), color = next(colorcycler) )
        plt.plot(convolveRL, label = str(param_name) + ' = ' + str(x), marker = next(markercycle), markevery= (next(markershiftcycle), 75), markersize = 20, linestyle = next(linecycler), color = next(colorcycler) )

    plt.legend(loc=5,prop={'size':30})
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid()

    my_axis = plt.gca()
    #my_axis.set_ylim(Variables.punishment-0.8, Variables.reward)
    my_axis.set_xlim(convolveSet, len(meansRL))
    
    plt.show()
        
#end of plotRewards method

def trainAgent(tries, episodes, scenario, explorationstrat, learning_alg, alpha, gamma, epsilon, sigma, delta, tau, suffix):

    filenameSteps = resultsFolder + 'stepsRL_' + str(suffix) + '.csv'
    filenameRewards = resultsFolder + 'rewardsRL_' + str(suffix) + '.csv'
        
    files.createFile(filenameSteps)
    files.createFile(filenameRewards)

    for i in range(tries):
        print('Training agent number: ' + str(i+1))
        agent = Agent(scenario, alpha, gamma, epsilon, sigma, delta, tau)
        [steps, rewards] = agent.train(episodes, explorationstrat, learning_alg)
        
        files.addToFile(filenameSteps, steps)
        files.addFloatToFile(filenameRewards, rewards)
    #endfor
        
    return agent
#end trainAgent method

if __name__ == "__main__":
    print("RL for cleaning a table is running ... ")
    tries = 20
    episodes = 1000
    explorationstrat = explorationstrat_epsgreedy
    learning_alg = learning_alg_sarsa
    alpha = 0.8  #0.8
    gamma = 0.9 #0.9
    epsilon = 0.1 #0.1
    sigma = 10 # 10
    delta = 0.1 #0.1
    tau = 0.001 #0.001

    #one flexible parameter for plotting, call trainAgent accordingly
    flex_param = [0.01, 0.05, 0.1, 0.15, 0.2]
    #just for display purposes in the plot, name of the flexible parameter
    param_name = 'epsilon'

    scenario = Scenario()

    for x in flex_param:
        print('RL is now training the agent')
        agent = trainAgent(tries, episodes, scenario, explorationstrat, learning_alg, alpha, gamma, x, sigma, delta, tau, x)

    plotRewards(flex_param, param_name, learning_alg, explorationstrat)
    
    print("The end")

# end of main method
