
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import linear_irl as linear_irl
import gridworld as gridworld
from mpl_toolkits.mplot3d import Axes3D

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
        
def main(grid_size, discount):
    """
    Run multi-agent linear programming inverse reinforcement learning on the gridworld MG.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MG discount factor. float.
    """

    
    play_num = 2

    gw = gridworld.Gridworld(play_num, grid_size, discount)
    act = np.array(gw.actions)
    
    policy_tu = [((0,1),(1,0)),((0,1),(0,1)),((0,1),(1,0)),((0,1),(0,-1)),\
            ((0,1),(0,1)),((-1,0),(0,1)),((-1,0),(-1,0)),((-1,0),(-1,0)),\
                ((-1,0),(0,1)),((-1,0),(0,1)),((-1,0),(0,1)),((-1,0),(0,1)),\
            ((-1,0),(1,0)),((-1,0),(-1,0)),((-1,0),(1,0)),((-1,0),(0,-1))]

   

    policy = np.zeros(gw.n_states,dtype = int)
    for i in range(gw.n_states):
        policy[i] = int(tu_action(act,policy_tu[i]))


    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])

    print(ground_r)

    
    r = linear_irl.irl(gw.n_players,gw.actions, gw.n_states, gw.n_actions, gw.transition_probability,policy, gw.discount, 10,0)

    print(r)

    a = ground_r[:,0].reshape((4,4))
    b = ground_r[:,1].reshape((4,4))
    a_1 = r[:16].reshape((4,4))
    b_1 = r[16:].reshape((4,4))

    print(b,b_1)
 
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    im1 = axes.flat[0].imshow(a)
    axes.flat[0].set_xlabel("B's square",fontsize=13)
    axes.flat[0].set_ylabel("A's square",fontsize=13)
    axes.flat[0].set_title("A's reward",fontsize=13)
    im2 = axes.flat[1].imshow(b)
    axes.flat[1].set_xlabel("B's square",fontsize=13)
    axes.flat[1].set_title("B's reward",fontsize=13)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
    fig.colorbar(im1, cax=cbar_ax)
    plt.show()
       
    plt.subplot(2,2,3)
    plt.pcolormesh(a_1)
    plt.colorbar()
    plt.title("a_1")
    plt.subplot(2,2,4)
    plt.pcolormesh(b_1)
    plt.colorbar()
    plt.title("b_1")
    plt.show()
    

    

if __name__ == '__main__':
    main(2, 0.2)
