"""LQR, iLQR and MPC."""

import numpy as np
import gym
from deeprl_hw6.arm_env import TwoLinkArmEnv
from controllers import *
import ipdb
import argparse
from copy import deepcopy

def run_lqr(env):
  print('Running LQR')
  present_state = env.reset()
  is_done = False
  prev_action = np.array([1.0,1.0])  #Let this be the action initialization
  num_steps = 0
  num_steps_threshold = 300
  total_rewards = 0
  while(not is_done and num_steps<num_steps_threshold):
    sim_env = deepcopy(env)
    num_steps += 1
    u = calc_lqr_input(env,sim_env,prev_action)
    print("Old state is",env.state)
    new_state,rewards,is_done,listo = env.step(u)
    print("New state is",new_state)
    present_state = new_state
    print(rewards)
    total_rewards+=rewards
    prev_action = u
    # print("Previous Action: ",prev_action)
  print('total_rewards are: ',total_rewards)
  print('Number of steps: ',num_steps)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='LQR Parser')
  parser.add_argument('--total_horizon',dest='T',type=int,default=100)
  args = parser.parse_args()
  env = gym.make('TwoLinkArm-v0')
  env.reset()
  run_lqr(env)
  # prev_action = np.array([0,0]) 
  # u = calc_lqr_input(env,np.copy(env),prev_action)



