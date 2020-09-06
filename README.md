# Halite(2020) Competition on [Kaggle](https://www.kaggle.com/c/halite/overview)

## Overview

Halite competition is a Deep Reinforcement learning competition where the agent should make decisions with the given environment. They are various possible situations where the agent will exposed to and should respond so that it maximizes its long-term benefit. They are some factors within the environment that one should take into account such as the ship's cargo, player's halite, cell's halite and etc.

## Agent's Decision process

There are six actions that a ship can take: mine, any of the four directions, and convert to a shipyard. Shipyard also have two actions that they can take either nothing or SPAWN (produce ship). Now the goal is to process the  object's surroundings and evaluate each of these actions with a weighting system so the taken decision would be as optimal as possible.