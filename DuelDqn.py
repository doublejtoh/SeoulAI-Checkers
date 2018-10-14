import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, session, input_height, input_width, name="main"):
        self.session = session
        self.input_height = input_height
        self.input_width = input_width
        self.net_name = name
        self._build_network()

    def _build_network(self, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(tf.float32, [None, self.input_height, self.input_width])

            X = tf.reshape(self.X, [-1, self.input_height, self.input_width, 1]) # 8X8X1

            self.conv1 = tf.contrib.layers.convolution2d(inputs=X, num_outputs=12, kernel_size=[4, 4], stride=[1, 1], padding='VALID', biases_initializer=None) # 5 X 5 X 12
            self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=24, kernel_size=[2,2], stride=[1,1], padding='VALID', biases_initializer=None) # 4 X 4 X 24

            self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=24, kernel_size=[2, 2], stride=[1,1], padding='VALID', biases_initializer=None) # 3 X 3 X 24

            self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=128, kernel_size=[2, 2], stride=[1, 1], padding='VALID', biases_initializer=None) # 2 X 2 X 128

            self.advantageStream, self.valueStream = tf.split(self.conv4, 2, 3) # split into 1 * 1 * 128
            self.advantageStream = tf.contrib.layers.flatten(self.advantageStream)
            self.valueStream = tf.contrib.layers.flatten(self.valueStream)

            self.advantage = tf.layers.dense(inputs=self.advantageStream, units=self.input_height * self.input_width, use_bias=False)
            self.value = tf.layers.dense(inputs=self.valueStream, units=1, use_bias=False)

            self.dest_Qout = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            # print(self.dest_Qout)
            # dest_reshaped_Qout = tf.reshape(self.dest_Qout,[self.input_height * self.input_width])
            # self.dest_Qpred_max_index =  ( int(dest_argmax / self.input_height), dest_argmax % self.input_height )# index of Max value according to Qout
            self.dest_Qtarget = tf.placeholder(tf.float32, [None, self.input_height * self.input_width])

            self.src_Qout = tf.layers.dense(inputs=self.valueStream, units=self.input_height * self.input_width, use_bias=False) # Predict according to Value stream(state stream).
            # src_reshaped_Qout = np.reshape(self.src_Qout, [-1, self.input_height * self.input_width])
            # self.src_Qpred_max_index = np.unravel_index(np.argmax(src_reshaped_Qout, axis=None), shape=src_reshaped_Qout.shape)

            self.loss = tf.reduce_mean(tf.square(self.dest_Qtarget - self.dest_Qout))
            self.train = tf.train.AdamOptimizer(l_rate).minimize(self.loss)

    def predict_dest(self, state, src_row=None, src_col=None, type=None):
        x = np.reshape(state, [1, self.input_height, self.input_width])
        if type == "all": # returns total array of dest Qout
            return self.session.run(self.dest_Qout, feed_dict={self.X: x})
        elif type == "max": # returns index of max value in dest_Qout.
            candidate_src_rows = [[src_row + 1, src_row + 1, src_row - 1, src_row - 1, ],
                                  [src_row + 2, src_row + 2, src_row - 2, src_row - 2]]  # 후보들. 대각선 후보들
            candidate_src_cols = [[src_col + 1, src_col - 1, src_col + 1, src_row - 1, ],
                                  [src_col + 2, src_col - 2, src_col + 2, src_col - 2]]  # 후보들. 대각선 후보들
            final_src_rows = []  # pre processing final src rows.
            final_src_cols = []  # pre processing final col rows.
            dest_Qout = self.session.run(self.dest_Qout, feed_dict={self.X: x})
            dest_Qout_reshaped = np.reshape(dest_Qout, [self.input_height, self.input_width])

            # possible row, col processing for step size 1.
            for i in range(len(candidate_src_rows[0])):
                if candidate_src_rows[0][i] >= 0 and candidate_src_rows[0][i] <= 7:
                    if state[candidate_src_rows[0][i], candidate_src_cols[0][i]] == 0:
                        final_src_rows.append(candidate_src_rows[0][i])
                        final_src_cols.append(candidate_src_cols[0][i])

            # possible row, col processing for step size 2.
            for i in range(len(candidate_src_rows[1])):
                if candidate_src_rows[1][i] >= 1 and candidate_src_cols[1][i] <= 6: # step size 2 인데 range넘어가면 안됨.
                    if state[candidate_src_rows[1][i], candidate_src_cols[1][i]] == 0: # step size 2일때 그곳에 아무도 없고,
                        if i == 0: # src_row+2, src_col+2 일경우
                            if state[candidate_src_rows[1][i] - 1, candidate_src_cols[1][i] - 1] == 20: # src_row+1, src_col+1에 상대 말이 있으면
                                final_src_rows.append(candidate_src_rows[1][i])
                                final_src_cols.append(candidate_src_cols[1][i])
                        elif i == 1:
                            if state[candidate_src_rows[1][i] - 1, candidate_src_cols[1][i] + 1] == 20: # src_row+1, src_col-1에 상대 말이 있으면
                                final_src_rows.append(candidate_src_rows[1][i])
                                final_src_cols.append(candidate_src_cols[1][i])
                        elif i == 2:
                            if state[candidate_src_rows[1][i] + 1, candidate_src_cols[1][i] - 1] == 20: # src_row-1, src_col+1에 상대 말이 있으면
                                final_src_rows.append(candidate_src_rows[1][i])
                                final_src_cols.append(candidate_src_cols[1][i])
                        elif i == 3:
                            if state[candidate_src_rows[1][i] + 1, candidate_src_cols[1][i] + 1] == 20: # src_row-1, src_col-1에 상대 말이 있으면
                                final_src_rows.append(candidate_src_rows[1][i])
                                final_src_cols.append(candidate_src_cols[1][i])

            if len(final_src_rows) == 0:
                return np.unravel_index(np.argmax(dest_Qout_reshaped, axis=None), dest_Qout_reshaped.shape)
            max = dest_Qout_reshaped[final_src_rows[0], final_src_cols[0]] # 미리 그냥 max 라고 설정해버림.

            for i in range (len(final_src_rows)):
                if dest_Qout_reshaped[final_src_rows[i], final_src_cols[i]] >= max:
                    max = dest_Qout_reshaped[final_src_rows[i], final_src_cols[i]]
                    max_value_row = final_src_rows[i]
                    max_value_col = final_src_cols[i]
            return (max_value_row, max_value_col)
        else:
            raise ValueError("Invalid Predict type")
    def predict_src(self, state, type):
        my_horse_row, my_horse_col = np.where(state == 10) # 내말의 위치. tuple array 리턴
        x = np.reshape(state, [1, self.input_height, self.input_width])
        if type == "all": # returns total array of src Qout
            return self.session.run(self.src_Qout, feed_dict={self.X: x})
        elif type == "max": # returns index of max value in src_Qout.
            src_Qout = self.session.run(self.src_Qout, feed_dict={self.X: x})
            src_Qout_reshaped = np.reshape(src_Qout, [self.input_height, self.input_width])
            max = src_Qout_reshaped[my_horse_row[0], my_horse_col[0]]
            for i in range(len(my_horse_row)):
                if src_Qout_reshaped[my_horse_row[i], my_horse_col[i]] >= max:
                    max = src_Qout_reshaped[my_horse_row[i], my_horse_col[i]]
                    max_value_row = my_horse_row[i]
                    max_value_col = my_horse_col[i]
            # print(src_Qpred_max_index)
            return (max_value_row, max_value_col)
        else:
            raise ValueError("Invalid Predict type")
    def update(self, x_stack, y_stack):
        return self.session.run([self.loss, self.train], feed_dict={self.X: x_stack, self.dest_Qtarget: np.reshape(y_stack, [-1, self.input_height * self.input_width])})


