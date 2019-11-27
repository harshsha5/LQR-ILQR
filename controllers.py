"""LQR, iLQR and MPC."""

import numpy as np
from scipy import linalg
import ipdb
from copy import deepcopy


def simulate_dynamics(env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    # print("In Simulate dynamics")
    previous_state = np.copy(x)
    env.state = np.copy(x)
    new_state,rewards,is_done,listo = env.step(u,dt)
    return np.reshape((new_state-previous_state)/dt, (-1,1))
    # return np.zeros(x.shape)


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    # print("In approximate_A")
    for i in range(x.shape[0]):
      perturbed_state = np.copy(x)
      perturbed_state[i] += delta
      derivative = simulate_dynamics(env,perturbed_state,u)
      if(i==0):
        A = derivative
      else:
        A = np.hstack((A,derivative))
    return A
    # return np.zeros((x.shape[0], x.shape[0]))


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    # print("In approximate_B")
    for i in range(u.shape[0]):
      perturbed_action = np.copy(u)
      perturbed_action[i] += delta
      derivative = simulate_dynamics(env,x,perturbed_action)
      if(i==0):
        B = derivative
      else:
        B = np.hstack((B,derivative))
    return B    
    # return np.zeros((x.shape[0], u.shape[0]))


def calc_lqr_input(env, sim_env, prev_action):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    # print('In calc_lqr_input')
    present_state = env.state
    ipdb.set_trace()
    A = approximate_A(deepcopy(sim_env),present_state,prev_action)
    B = approximate_B(sim_env,present_state,prev_action)
    P = linalg.solve_continuous_are(A, B, env.Q, env.R)
    u = -1*np.matmul(np.matmul(np.linalg.inv(env.R),np.matmul(B.T, P)),(present_state - env.goal).T)
    return u

    # return np.ones((2,))
