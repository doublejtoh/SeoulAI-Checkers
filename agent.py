from seoulai_gym.envs.checkers.agents import Agent
import tensorflow as tf
import numpy as np
from DuelDqn import DQN
from typing import List
from typing import Tuple
import random
from collections import deque


dis = 0.99
MAX_MEMORY = 50000

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.input_height, mainDQN.input_width)
    y_stack = np.empty(0).reshape(0, mainDQN.input_height, mainDQN.input_width)
    for state, action, reward, next_state, done in train_batch:
        state = np.reshape(state, [-1, mainDQN.input_height, mainDQN.input_width])
        Q = np.reshape(mainDQN.predict_dest(state,None, None, "all"), [-1, mainDQN.input_height, mainDQN.input_width])
        action_xpos, action_ypos = action
        if done:
            Q[0,action_xpos, action_ypos] = reward

        else:
            Q[0,action_xpos, action_ypos] = reward + dis * np.max(targetDQN.predict_dest(next_state, None, None, "all"))

        # print(y_stack.shape, Q.shape, state.shape)
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    return op_holder

class DqnAgent(Agent):
    def __init__(self, sess, name, ptype: int):
        super().__init__(name, ptype)
        self.sess = sess
        self.mainDqn = DQN(self.sess, 8, 8, name="main") # Q Net
        self.targetDqn = DQN(self.sess, 8, 8, name="target") # Q Target Net
        self.replay_memory = deque()



    def act(self, board: List[List], episode:  int) -> Tuple[int, int, int, int]:
        # Get what horse to move from mainDQN.
        # print(self.mainDqn.predict_src(board, "max"))
        from_row, from_col = self.mainDqn.predict_src(board, "max")

        # Get Where to move from mainDQN.
        e = 1. / ((episode / 10) + 1)
        if np.random.rand(1) < e:
            to_row = random.choice(np.arange(8))
            to_col = random.choice(np.arange(8))
        else:
            to_row, to_col = self.mainDqn.predict_dest(board, from_row, from_col, "max")

        return from_row, from_col, to_row, to_col

    def consume(self, obs: List[List], action: Tuple[int, int], reward: float, done: bool, episode: int, next_state: List[List]) -> None:
        # Replay Train.
        self.replay_memory.append((obs, action, reward, next_state, done))
        # print("len: ", len(self.replay_memory))
        if (len(self.replay_memory) > MAX_MEMORY):
            self.replay_memory.popleft()

    def consume_after_episode(self, episode):
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        if episode % 10 == 1:
            for _ in range(50):
                # print(len(self.replay_memory))
                minibatch = random.sample(self.replay_memory, 10)
                loss, _ = replay_train(self.mainDqn, self.targetDqn, minibatch)
            print("loss: ", loss)
            self.sess.run(copy_ops)