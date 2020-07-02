# Project 3 - Udacity Tennis Environment
## Challenge
The environment to solve involves training two agents to bounce a ball over a net for as long as possible. The aim of 
each agent is to keep the ball in play without it hitting the ground or going out of bounds.

A reward of +0.1 is given to the agent if it hits the ball over the net. If the agent lets the ball fall or hits it 
out it receives a reward of -0.01. The task is episodic and the environment is considered solved 
when an average score of +0.5 is achieved over 100 consecutive episodes.

The state space has 24 variables. It contains the ball's velocity and position as well as the agent's racket position.
Each agent receives its own observation of the environment.

For the action space, each agent has two continuous actions available. The first action corresponds to movement towards 
or away from the net. The second corresponds to a jump, i.e. moving vertically. Both actions are continuous and an 
agent action consists of a value for both actions at each time step.

## Development environment
+ This agent has been trained using __Python 3.6.8__ on __Windows 10__
+ The __requirements.txt__ file contains all required Python packages
+ To install these packages, navigate to your Python virtual 
environment, activate it and use: 
     - pip install -r requirements.txt 

The Tennis environment is a modified version of the one 
provided by [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). 
However it is not the same and for this project the modified version 
provided by Udacity has been used for training.

__Ensure__ the location of the Udacity modified Tennis 
environment folder is in the same folder as the tennis_colaboration.py 
file. This will make sure the code finds the environment.

## Running code
tennis_collaboration.py contains the entry point to the code, whether for 
training a new agent or for viewing a previously trained agent.
#### Training
To run the code and train a new agent, run tennis_collaboration.py. This script 
will instantiate a Unity Environment and an Agent class and pass these as 
arguments to the "train_maddpg" function. This function loops through episodes 
and passes the states, actions and rewards to the agent for training. 
Make sure the Unity Environment is given the correct argument for the 
file_name parameter, which should be the the file path, file name and 
extension of the Unity Environment.

To run (for training):
* Check file_name parameter for Unity Environment
* Check the hyperparameters given to the Agent class
* Call the "train_maddpg" method with the Agent and the Unity environment as 
arguments
#### Demonstrating
To run the code and view the trained agent in action in its environment, 
make an edit to tennis_collaboration.py at lines 110 - 112, to comment out the 
"train_maddpg" function and to call the "demonstrate_multi_agent" function. Again, 
make sure the Unity Environment has the correct file_name argument as well as having a 
correctly defined agent. The checkpoint files with the best 
weights for the multi agent has been provided as part of this submission under the four *_checkpoint.pth files.

To run (for demonstration):
* Check file_name parameter for Unity Environment
* Check the agent is instantiated correctly
* Call the "demonstrate_multi_agent" method with the Unity environment and the 
agent as arguments
