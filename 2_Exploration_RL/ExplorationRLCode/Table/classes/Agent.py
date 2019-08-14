# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 17:43:35 2015

@author: cruz
"""

import numpy as np
import Variables

class Agent(object):

    def __init__(self, scenario, alpha, gamma, epsilon, sigma, delta, tau):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma = sigma
        self.delta = delta
        self.tau = tau
        self.scenario = scenario
        self.numberOfStates = self.scenario.getNumberOfStates()
        self.numberOfActions = self.scenario.getNumberOfActions()
        self.Q = np.random.uniform(0.0,0.01,(self.numberOfStates,self.numberOfActions))
        self.vdbe_eps = np.empty(self.numberOfStates)
        self.vdbe_eps.fill(self.epsilon)
        self.feedbackAmount = 0
    #end of __init__ method

    def softmax(self, q):
        """Compute softmax values for q."""
        sf = np.exp(q/self.tau)
        sf = sf/np.sum(sf, axis=0)
        return sf

    def selectAction(self, state, explorationstrat):
        if explorationstrat == Variables.explorationstrat_epsgreedy:
            if (np.random.rand() <= self.epsilon):
                action = np.random.randint(self.numberOfActions)
            else:
                action = np.argmax(self.Q[state,:])
        elif explorationstrat == Variables.explorationstrat_softmax:
            prob_q = self.softmax(self.Q[state, :])
            total_prob_q = np.cumsum(prob_q)
            sample = np.random.rand()
            action = np.searchsorted(total_prob_q, sample)
        elif explorationstrat == Variables.explorationstrat_vdbe:
            if (np.random.rand() <= self.vdbe_eps[state]):
                action = np.random.randint(self.numberOfActions)
            else:
                action = np.argmax(self.Q[state,:])
        elif explorationstrat == Variables.explorationstrat_vdbe_softmax:
            if (np.random.rand() <= self.vdbe_eps[state]):
                prob_q = self.softmax(self.Q[state, :])
                total_prob_q = np.cumsum(prob_q)
                sample = np.random.rand()
                action = np.searchsorted(total_prob_q, sample)
            else:
                action = np.argmax(self.Q[state,:])
        else:
            test = 1#todo error
        return action
    #end of selectAction method

    def train(self, episodes, explorationstrat, learning_alg):
        contCatastrophic = 0
        contFinalReached = 0
        steps = np.zeros(episodes)
        rewards = np.zeros(episodes)
        
        for i in range(episodes):
            contSteps = 0
            accReward = 0
            self.scenario.resetScenario()
            state = self.scenario.getState()


            #expisode
            while True:
                #choose action
                action = self.selectAction(state, explorationstrat)
                #perform action
                self.scenario.executeAction(action)
                contSteps += 1

                #get reward
                reward = self.scenario.getReward()
                accReward += reward
                #catastrophic state

                stateNew = self.scenario.getState()

                if reward == Variables.punishment:
                    contCatastrophic += 1
                    self.Q[state,action] = -0.1
                    break

                if learning_alg == Variables.learning_alg_sarsa:
                    actionNew = self.selectAction(stateNew, explorationstrat)
                elif learning_alg == Variables.learning_alg_qlearning:
                    actionNew = np.argmax(self.Q[stateNew,:])

                # updating Q-values
                new_Q = self.Q[state, action] + self.alpha * (reward + self.gamma *
                                                self.Q[stateNew,actionNew] -
                                                self.Q[state,action])

                if reward == Variables.reward:
                    contFinalReached += 1
                    break

                #update state-dependent epsilon
                if (explorationstrat == Variables.explorationstrat_vdbe) or (explorationstrat == Variables.explorationstrat_vdbe_softmax):
                    updatevalue = (1 - np.exp((-np.abs(new_Q - self.Q[state, action]))/self.sigma)) / (1 + np.exp((-np.abs(new_Q - self.Q[state, action]))/self.sigma))
                    self.vdbe_eps[state] = self.delta * updatevalue + (1- self.delta) * self.vdbe_eps[state]




                self.Q[state, action] = new_Q
                state = stateNew
            #end of while
            steps[i] = contSteps
            rewards[i]=accReward
        #end of for
        return steps,rewards
    #end of train method

#end of class Agent
