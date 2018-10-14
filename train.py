import seoulai_gym as gym
import tensorflow as tf
from seoulai_gym.envs.checkers.utils import board_list2numpy
from seoulai_gym.envs.checkers.agents import RandomAgentLight
from seoulai_gym.envs.checkers.base import Constants
from agent import DqnAgent

env = gym.make("Checkers")

####### GAME SETTING  #######
checkers_height = 8
checkers_width = 8


####### H PARAMS      #######
dis = 0.99

####### Model Restore, Save #######

save_file = './train_model.ckpt'

def main():
    max_episodes = 10000
    with tf.Session() as sess:
        # saver = tf.train.Saver()
        ####### Agent Setting #######
        MasterAgent = RandomAgentLight("Teacher Agent")
        MyAgent = DqnAgent(sess, "doublejtoh Agent", Constants().LIGHT)

        tf.global_variables_initializer().run()

        current_agent = MasterAgent
        next_agent = MyAgent

        for episode in range(max_episodes):
            step = 0
            done = False
            obs = env.reset()
            env.render()
            while not done:
                state = board_list2numpy(obs)
                if current_agent._name == 'doublejtoh Agent':
                    from_row, from_col, to_row, to_col = current_agent.act(state, episode)
                    # print(from_row, from_col, to_row, to_col)
                else:
                    from_row, from_col, to_row, to_col = current_agent.act(obs)
                    # print("Teacher Agent: ", from_row, from_col, to_row, to_col)
                obs, reward, done, info = env.step(current_agent, from_row, from_col, to_row, to_col)
                # print(info)
                if done:
                    print(f"{current_agent} agent wins.")
                action = (to_row, to_col)
                next_state = board_list2numpy(obs)
                if current_agent._name == 'doublejtoh Agent':
                    current_agent.consume(state, action, reward, done, episode, next_state) # here, obs means new state. save to memory.

                ### change turn ###
                temporary_agent = current_agent
                current_agent = next_agent
                next_agent = temporary_agent


                env.render()

                step += 1
            print("Episode ", episode, "step: ", step)

            # if current_agent._name == 'doublejtoh Agent':
            #     current_agent.consume_after_episode(episode) # replay train.
            MyAgent.consume_after_episode(episode)

        # if episode % 1000 == 0: # save to model every 1000 episodes.
        #     saver = tf.train.Saver()
        #     saver.save(sess, save_file)
        #     print("trained Model saved: check ", save_file)



if __name__ == "__main__":
    main()
