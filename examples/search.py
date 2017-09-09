import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9
#训练回合数
MAX_EPISODES = 13
FRESH_TIME = 0.3
#创建 q 表
def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return table
#根据当前状态选择动作
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name
#与环境交互
def get_env_feedback(S, A):
    if A == 'right':
        R = 1
        if S == 5:
            S_ = 'termination'
        else:
            S_ = S + 1
    else:
        R = -1
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R
#更新环境
def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES - 1) + ['T']
    if S == 'termination':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print()
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ != 'termination':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else :
                q_target = R
                is_terminated = True
            q_table.ix[S, A] = (1 - ALPHA) * q_predict + ALPHA * q_target
            S = S_
            update_env(S, episode, step_counter+1)
            step_counter += 1
            # print(q_table)
        print(q_table)
    return q_table

if __name__ == '__main__':
    # table = build_q_table(N_STATES, ACTIONS)
    rl()