"""LQR, iLQR and MPC."""

from controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
from copy import deepcopy
import ipdb
import matplotlib.pyplot as plt


def simulate_dynamics_next(env, x, u):
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

    Returns
    -------
    next_x: np.array
    """
    return np.zeros(x.shape)


def cost_inter(env, x, u):
    """intermediate cost function

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

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """

    l = np.sum(u**2)
    l_x = np.zeros(x.shape[0])  #As l is independent of x
    l_xx = np.zeros((x.shape[0],x.shape[0])) #As l is independent of x
    l_u = 2*u
    l_uu = 2 * np.eye(env.DOF)
    l_ux = np.zeros((env.DOF, x.shape[0]))

    return l,l_x,l_xx,l_u,l_uu,l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """

    pos_err = x - env.goal
    l = (10**4) * np.sum(pos_err**2)
    # l_x = 2*(10**4) * np.sum(abs(pos_err))             #TODO no abs
    l_x = 2*(10**4) * (pos_err)   
    l_xx = 2*(10**4) * np.eye(x.shape[0])                                     #TODO
    return l,l_x,l_xx

def simulate(env, x0, U):
    time_steps = U.shape[0]
    X_new = np.tile(x0, (time_steps+1, 1))
    trajectory_cost = 0

    for i in range(time_steps):
      X_new[i+1],_,_,_ = env.step(U[i])
      step_cost, _, _, _, _, _ = cost_inter(deepcopy(env),X_new[i+1],U[i])
      trajectory_cost+=step_cost   #Do we need a dt here?

    final_cost,_,_ = cost_final(deepcopy(env),X_new[-1])
    trajectory_cost*=10**(-3)
    trajectory_cost+= final_cost 
    print("Intermediate cost ",trajectory_cost-final_cost," \t Final Cost ",final_cost)

    return trajectory_cost,X_new

def plot_costs(cost_array):
  t = np.arange(start = 1, stop = len(cost_array)+1, step = 1)
  plt.plot(t, np.asarray(cost_array), 'g',label="Total Cost") 
  plt.ylabel('Total Cost')
  plt.xlabel('Iteration Number')
  plt.legend(loc='upper right')
  plt.title('Total Cost VS Iteration Number')
  plt.show()  

def inv_stable(M, lamb=1):
    """Inverts matrix M in a numerically stable manner.

    This involves looking at the eigenvector (i.e., spectral) decomposition of the
    matrix, and (1) removing any eigenvectors with non-positive eingenvalues, and
    (2) adding a constant to all eigenvalues.
    """
    M_evals, M_evecs = np.linalg.eig(M)
    M_evals[M_evals < 0] = 0.0
    M_evals += lamb
    M_inv = np.dot(M_evecs,
                   np.dot(np.diag(1.0 / M_evals), M_evecs.T))
    return M_inv

def calc_ilqr_input(env, sim_env, previous_actions, tN=100, max_iter=1e6):
    """Calculate the optimal control input for the given state.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """

    CONVERGENCE_THRESHOLD = 0.0001
    LAMB = 1
    LAMB_FACTOR = 2                             #Note by Harsh: Code referred https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
    cost_array = []

    if(previous_actions is None):
        prev_single_action = np.full((env.DOF, ), 0.0)
        previous_actions = np.tile(prev_single_action, (tN, 1))

    env.reset()
    present_state = np.copy(env.state)
    old_trajectory_cost, X_traj = simulate(deepcopy(env),np.copy(present_state),previous_actions)
    cost_array = [old_trajectory_cost]

    for iteration_count in range(int(max_iter)):
      env.reset()
      present_state = np.copy(env.state)

      f_x = np.zeros((tN, present_state.shape[0], present_state.shape[0])) 
      f_u = np.zeros((tN, present_state.shape[0], env.DOF))
      l = np.zeros((tN+1,1))                                    #See if the +1 is correct or not, since the last step won't have a control step but would depend on the cpst final
      l_x = np.zeros((tN+1, present_state.shape[0]))
      l_xx = np.zeros((tN+1, present_state.shape[0],present_state.shape[0]))
      l_u = np.zeros((tN, env.DOF))
      l_uu = np.zeros((tN, env.DOF,env.DOF))
      l_ux = np.zeros((tN, env.DOF,present_state.shape[0]))

      for i in range(tN):
        A = approximate_A(sim_env,present_state,previous_actions[i])
        B = approximate_B(sim_env,present_state,previous_actions[i])
        f_x[i,:,:] = A*env.dt + np.eye(present_state.shape[0])
        f_u[i,:,:] = B*env.dt                                        #See if this A and B is correct or if we need to follow the dt approach
        l[i], l_x[i,:], l_xx[i,:,:], l_u[i,:], l_uu[i,:,:], l_ux[i,:,:] = cost_inter(env,present_state,previous_actions[i])
        l[i]*=env.dt
        l_x[i,:]*=env.dt
        l_xx[i,:,:]*=env.dt
        l_u[i,:]*=env.dt
        l_uu[i,:,:]*=env.dt
        l_ux[i,:,:]*=env.dt
        present_state,_,done,_ = env.step(previous_actions[i])   #See how to handle done situation here
        if(done):
          print("LQR has converged")
          pres_state = sim_env.reset()
          old_trajectory_cost, state_list = simulate(deepcopy(sim_env),np.copy(pres_state),previous_actions)
          return previous_actions,state_list

      l[-1],l_x[-1],l_xx[-1] = cost_final(env,present_state)

      V = np.copy(l[-1]) 
      V_x = np.copy(l_x[-1])
      V_xx = np.copy(l_xx[-1]) 
      k = np.zeros((tN, env.DOF)) 
      K = np.zeros((tN, env.DOF, present_state.shape[0])) 

      for t in range(tN-1, -1, -1):
        Q_x = l_x[t] + np.dot(f_x[t].T, V_x)
        Q_u = l_u[t] + np.dot(f_u[t].T, V_x)
        Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
        Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
        Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))
        # Q_uu_inv = inv_stable(Q_uu,LAMB)
        Q_uu_inv = np.linalg.pinv(Q_uu + LAMB*np.eye(Q_uu.shape[0]))       #TODO
        # Q_uu_inv = np.linalg.pinv(Q_uu) 

        k[t] = -np.dot(Q_uu_inv, Q_u)
        K[t] = -np.dot(Q_uu_inv, Q_ux)

        V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
        V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

      new_actions = np.zeros((tN, env.DOF))
      env.reset()
      start_state = np.copy(env.state)
      new_state = np.copy(env.state) 
      for t in range(tN): 
        new_actions[t] = previous_actions[t] + k[t] + np.dot(K[t], new_state - X_traj[t]) 
        new_state,_,_,_ = env.step(new_actions[t]) 

      env.reset()
      new_trajectory_cost, Xnew  = simulate(deepcopy(env),np.copy(start_state),new_actions)
      # print("New Trajectory cost is: ",new_trajectory_cost)

      if(abs(old_trajectory_cost-new_trajectory_cost)<CONVERGENCE_THRESHOLD):
        print("Convergence achived in iteration_count ",iteration_count)
        plot_costs(cost_array)
        return new_actions,Xnew
      if(old_trajectory_cost-new_trajectory_cost>0):
        cost_array.append(new_trajectory_cost)
        previous_actions = np.copy(new_actions)
        old_trajectory_cost, X_traj = new_trajectory_cost,Xnew
        LAMB /= LAMB_FACTOR
        if(LAMB<0.0001):
          LAMB = 0.0001
      else:
        LAMB *= LAMB_FACTOR
        if(LAMB>10000):
          print("Convergence Failed")
          break

    # return np.zeros((50, 2))
