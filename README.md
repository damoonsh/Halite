# Halite(2020) Competition on [Kaggle](https://www.kaggle.com/c/halite/overview)

## Overview

Halite competition is a deep reinforcement learning competition where the agent should make decisions with the given environment. They are various possible situations where the agent will be in and should respond in a way that suits its environment. They are some factors within the environment that one should take into account.

## Agent's Decision process

There are six actions that a ship can take: mine, any of the four directions, and convert to a shipyard. Shipyard also have two actions that they can take either nothing or SPAWN (produce ship). Now the goal is to process the  object's surroundings and evaluate each of these actions with a weighting system so the taken decision would be as optimal as possible.