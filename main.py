"""LQR, iLQR and MPC."""

import numpy as np
import gym
from deeprl_hw6.arm_env import TwoLinkArmEnv
from controllers import *
from ilqr import *
import ipdb
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt

def plot_graphs(t_state,t_action,q1,q2,q1_dot,q2_dot,u1,u2):
  plt.plot(t_state, q1, 'r',label="q1")  
  plt.plot(t_state, q2, 'g',label="q2") 
  plt.legend(loc='upper right')
  plt.ylabel('Joint Angles')
  plt.xlabel('Step Number')
  plt.title('Joint Angles VS Steps')
  plt.show()
  plt.plot(t_state, q1_dot, 'r',label="q1_dot") 
  plt.plot(t_state, q2_dot, 'g',label="q2_dot") 
  plt.ylabel('Joint Velocities')
  plt.xlabel('Step Number')
  plt.title('Joint Velocities VS Steps')
  plt.legend(loc='upper right')
  plt.show()
  plt.plot(t_action, u1, 'b',label="u1") 
  plt.plot(t_action, u2, 'g',label="u2") 
  plt.ylabel('Action')
  plt.xlabel('Step Number')
  plt.legend(loc='upper right')
  plt.title('Action VS Steps')
  plt.show()  

def run_lqr(env):
  print('Running LQR')
  present_state = env.reset()
  is_done = False
  prev_action = np.array([1.0,1.0])  #Let this be the action initialization
  num_steps = 0
  num_steps_threshold = 300
  total_rewards = 0
  action_list = []
  state_list = []
  while(not is_done and num_steps<num_steps_threshold):
    # env.render()
    sim_env = deepcopy(env)
    state_list.append(env.state)
    num_steps += 1
    u = calc_lqr_input(env,sim_env,prev_action)
    action_list.append(u)
    # print("Old state is",env.state)
    new_state,rewards,is_done,listo = env.step(u)
    # print("New state is",new_state)
    present_state = new_state
    print(rewards)
    total_rewards+=rewards
    # prev_action = u
    # print("Previous Action: ",prev_action)
  print('total_rewards are: ',total_rewards)
  print('Number of steps: ',num_steps)

  t = np.arange(start = 1, stop = num_steps+1, step = 1)
  q1 = [item[0] for item in state_list]
  q2 = [item[1] for item in state_list]
  q1_dot = [item[2] for item in state_list]
  q2_dot = [item[3] for item in state_list]
  u1 = [item[0] for item in action_list]
  u2 = [item[1] for item in action_list]

  # plot_graphs(t,t,q1,q2,q1_dot,q2_dot,u1,u2)

  return action_list

def run_ilqr(env,initialize_with_LQR = False,total_horizon=50):
  print('Running iLQR')
  present_state = env.reset()
  is_done = False
  if(initialize_with_LQR):
    prev_action_list = run_lqr(deepcopy(env))
    prev_action_list = np.asarray(prev_action_list[:50])
  else:
    prev_action_list = None
  num_steps = 0
  total_rewards = 0
  action_list = []
  state_list = []
  new_actions,X_new = calc_ilqr_input(env,deepcopy(env),prev_action_list)

  st = env.reset()
  for i in range(new_actions.shape[0]):
    st,step_reward,is_done,_ = env.step(new_actions[i])
    total_rewards+=step_reward
    num_steps+=1
    if(is_done):
      break

  print('total_rewards are: ',total_rewards)
  print('Number of steps: ',num_steps)

  q1 = X_new[:,0]
  q2 = X_new[:,1]
  q1_dot = X_new[:,2]
  q2_dot = X_new[:,3]
  u1 = new_actions[:,0]
  u2 = new_actions[:,1]
  t_state = np.arange(start = 1, stop = q1.shape[0]+1, step = 1)
  t_action = np.arange(start = 1, stop = u1.shape[0]+1, step = 1)
  plot_graphs(t_state,t_action,q1,q2,q1_dot,q2_dot,u1,u2)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='LQR Parser')
  parser.add_argument('--total_horizon',dest='T',type=int,default=50)
  args = parser.parse_args()
  env = gym.make('TwoLinkArm-v0')
  env.reset()
  total_horizon = args.T
  run_lqr(env)
  # run_ilqr(env,False)
  # prev_action = np.array([0,0]) 
  # u = calc_lqr_input(env,np.copy(env),prev_action)



