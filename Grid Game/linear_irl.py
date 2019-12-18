
import random

import numpy as np
from cvxopt import matrix, solvers

def irl(player_num,actions, n_states, n_actions, transition_probability, policy, discount, Rmax,
        l1):
    """
    
    player_num : the number of players. int.
    actions : the set of all joint actions
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    policy: Vector mapping state ints to action ints. Shape (N,).
    discount: Discount factor. float.
    Rmax: Maximum reward. float.
    l1: l1 regularisation. float.
    -> Reward vector
    """
    
    A = {(0,1),(1,0),(0,-1),(-1,0)}
    # The transition policy convention is different here to the rest of the code
    # for legacy reasons; here, we reorder axes to fix this. We expect the
    # new probabilities to be of the shape (A, N, N).
    transition_probability = np.transpose(transition_probability, (1, 0, 2))

    def tu_action(act,tup):
        """
        convert actions presented by in tuple into the corresponding actions int
        act: the set of all actions tuple
        tup: tuple of joint action((x1,y1),(x2,y2))

        -> the joint action int of the set 
        """
        for i in range(len(act)):
            if(act[i]==np.array(tup)).all():
                return i
        return ValueError("Unexpected act.")

    
    


    
    def cal_set(a,i,acts):
        """
        calculate a set of actions where the ith player in joint action a changes his action
        a: action int
        acts: the set of all joint actions
        i: player int
        -> a set of actions 
        """
        a = int(a)
        a_tup = acts[a]
        a_tup_i = a_tup[i]
        
        res_set = set()
        
        rests = A - {a_tup_i}
        jointact = np.array(acts)
        a_array = jointact[a].copy()

        for rest in rests:
            a_new_array = a_array.copy()
            a_new_array[i] = np.array(rest)
            
            a_new = int(tu_action(jointact,a_new_array))
            
            res_set = res_set | {a_new}
        
        return res_set

    def T(a, s):
        """
        Shorthand for a dot product used a lot in the LP formulation.
        """
        T = np.dot(transition_probability[policy[s], s] -  transition_probability[a, s], np.linalg.inv(np.eye(n_states) - discount*transition_probability[policy[s]]))
        
        return T


    
    # Minimise c . x.
    c = -np.hstack([np.zeros(player_num*n_states), np.ones(player_num*n_states),-l1*np.ones(player_num*n_states)])
    zero_stack1 = np.zeros((n_states*(4-1), n_states))
    T_stack_0 = np.vstack([
        -T(a, s)
        for s in range(n_states)   
        for a in cal_set(policy[s],0,actions)
    ])

    T_stack_1 = np.vstack([
        -T(a, s)
        for s in range(n_states)   
        for a in cal_set(policy[s],1,actions)
    ])

    I_stack = np.vstack([
        np.eye(1, n_states, s)
        for s in range(n_states)
        for a in cal_set(policy[s],0,actions)
    ])

    I_stack2 = np.eye(n_states)
    zero_stack2 = np.zeros((n_states,n_states))
    print(zero_stack1.shape,T_stack_0.shape,T_stack_1.shape,I_stack.shape,I_stack2.shape)
   
    D = np.bmat([[T_stack_0, zero_stack1, I_stack, zero_stack1, zero_stack1, zero_stack1],    # -TR <= -t
                [zero_stack1, T_stack_1, zero_stack1, I_stack, zero_stack1, zero_stack1],      
                [T_stack_0, zero_stack1, zero_stack1, zero_stack1, zero_stack1, zero_stack1], # -TR <= 0
                [zero_stack1, T_stack_1, zero_stack1, zero_stack1, zero_stack1, zero_stack1], 
                [-I_stack2,zero_stack2, zero_stack2,zero_stack2, -I_stack2,zero_stack2],  # -R <= u
                [zero_stack2,-I_stack2,zero_stack2, zero_stack2,zero_stack2, -I_stack2], 
                [I_stack2,zero_stack2, zero_stack2,zero_stack2, -I_stack2,zero_stack2],  # R <= u
                [zero_stack2,I_stack2,zero_stack2, zero_stack2,zero_stack2, -I_stack2], 
                [-I_stack2,zero_stack2, zero_stack2,zero_stack2, zero_stack2,zero_stack2],  # -R <= Rmax 
                [zero_stack2,-I_stack2,zero_stack2, zero_stack2,zero_stack2, zero_stack2],  
                [I_stack2,zero_stack2, zero_stack2,zero_stack2, zero_stack2,zero_stack2],  # -R <= Rmax 
                [zero_stack2,I_stack2,zero_stack2, zero_stack2,zero_stack2, zero_stack2],
                ])

    b = np.zeros((player_num*n_states*(4-1)*2 +player_num*2*n_states, 1))


    b_bounds = np.vstack([Rmax*np.ones((player_num*n_states, 1))]*2)
    
    b = np.vstack((b, b_bounds))
    print(D.shape,b.shape,c.shape)
    A_ub = matrix(D)
    b = matrix(b)
    c = matrix(c)
    
    results = solvers.lp(c, A_ub, b)
    r = np.asarray(results["x"][:player_num*n_states], dtype=np.double)

    return r

