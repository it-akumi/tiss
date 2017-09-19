# coding: utf-8

import numpy as np
from chainer import cuda
from scipy.spatial.distance import cosine

from config.log import APP_KEY
import logging
app_logger = logging.getLogger(APP_KEY)


class Experience:
    def __init__(self, use_gpu=0, data_size=10**5, replay_size=32, hist_size=1, initial_exploration=10**3, dim=10240):

        self.use_gpu = use_gpu
        self.data_size = data_size
        self.preplay_data_size = 10000
        self.replay_size = replay_size
        self.preplay_size = 5
        self.hist_size = hist_size
        # self.initial_exploration = 10
        self.initial_exploration = initial_exploration
        self.dim = dim

        self.d = [np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

        self.preplay_d = [np.zeros((self.preplay_data_size, self.hist_size, self.dim), dtype=np.uint8),
                          np.zeros(self.preplay_data_size, dtype=np.uint8),
                          np.zeros((self.preplay_data_size, 1), dtype=np.int8),
                          np.zeros((self.preplay_data_size, self.hist_size, self.dim), dtype=np.uint8),
                          np.zeros((self.preplay_data_size, 1), dtype=np.bool)]


    def stock(self, time, state, action, reward, state_dash, episode_end_flag):
        data_index = time % self.data_size
        preplay_data_index = time % self.preplay_data_size

        if episode_end_flag is True:
            self.d[0][data_index] = np.array(state, dtype=np.float32)
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
        else:
            self.d[0][data_index] = np.array(state, dtype=np.float32)
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
            self.d[3][data_index] = np.array(state_dash, dtype=np.float32)
        self.d[4][data_index] = episode_end_flag

        '''
        if episode_end_flag is True:
            self.preplay_d[0][preplay_data_index] = np.array(state, dtype=np.float32)
            self.preplay_d[1][preplay_data_index] = action
            self.preplay_d[2][preplay_data_index] = reward
        else:
            self.preplay_d[0][preplay_data_index] = np.array(state, dtype=np.float32)
            self.preplay_d[1][preplay_data_index] = action
            self.preplay_d[2][preplay_data_index] = reward
            self.preplay_d[3][preplay_data_index] = np.array(state_dash, dtype=np.float32)
        self.preplay_d[4][preplay_data_index] = episode_end_flag
        '''

    def replay(self, time):
        replay_start = False
        if self.initial_exploration < time:
            replay_start = True
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.d[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.d[1][replay_index[i]]
                r_replay[i] = self.d[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.d[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.d[4][replay_index[i]]

            if self.use_gpu >= 0:
                s_replay = cuda.to_gpu(s_replay)
                s_dash_replay = cuda.to_gpu(s_dash_replay)

            return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay

        else:
            return replay_start, 0, 0, 0, 0, False

    def preplay(self, time):
        preplay_start = False
        if self.initial_exploration < time:
            preplay_start = True
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                preplay_index = np.random.randint(0, time, (self.preplay_size, 1))
            else:
                preplay_index = np.random.randint(0, self.data_size, (self.preplay_size, 1))

            a_all = np.array([0, 1, 2], dtype=np.uint8)

            s_preplay = np.ndarray(shape=(self.preplay_size, self.hist_size, self.dim), dtype=np.float32)
            a_preplay = np.ndarray(shape=(self.preplay_size, 1), dtype=np.uint8)
            r_preplay = np.ndarray(shape=(self.preplay_size, 1), dtype=np.float32)
            s_dash_preplay = np.ndarray(shape=(self.preplay_size, self.hist_size, self.dim), dtype=np.float32)
            episode_end_preplay = np.ndarray(shape=(self.preplay_size, 1), dtype=np.bool)

            all_choice = []
            if time < self.data_size:
                all_choice = np.arange(time)
            else:
                all_choice = np.arange(self.data_size)
            np.random.shuffle(all_choice)
            choice = all_choice[:self.preplay_data_size]

            for (i, c) in enumerate(choice):
                self.preplay_d[0][i] = self.d[0][c]
                self.preplay_d[1][i] = self.d[1][c]
                self.preplay_d[2][i] = self.d[2][c]
                self.preplay_d[3][i] = self.d[3][c]
                self.preplay_d[4][i] = self.d[4][c]


            for i in xrange(self.preplay_size):
                s_preplay[i] = np.asarray(self.d[0][preplay_index[i]], dtype=np.float32)

                not_a = self.d[1][preplay_index[i]]
                a_preplay[i] = np.delete(a_all, not_a)[np.random.randint(0, 2)]

                if self.use_gpu >= 0:
                    similar_state_index = self._similar_state_index_norm_gpu(s_preplay[i], time)
                    # similar_state_index = self._similar_state_index_cos_gpu(s_preplay[i], time)
                else:
                    # similar_state_index = self._similar_state_index_norm(s_preplay[i], time)
                    similar_state_index = self._similar_state_index_cos(s_preplay[i], time)

                r_preplay[i] = self.preplay_d[2][similar_state_index]
                s_dash_preplay[i] = np.array(self.preplay_d[3][similar_state_index], dtype=np.float32)
                episode_end_preplay[i] = self.preplay_d[4][similar_state_index]

            if self.use_gpu >= 0:
                s_preplay = cuda.to_gpu(s_preplay)
                s_dash_preplay = cuda.to_gpu(s_dash_preplay)

            return preplay_start, s_preplay, a_preplay, r_preplay, s_dash_preplay, episode_end_preplay

        else:
            return preplay_start, 0, 0, 0, 0, False

    def _similar_state_index_norm(self, state, time):
        epsilon = 0.000001
        state_all = []
        if time < self.preplay_data_size:  # during the first sweep of the History Data
            state_all = np.array(self.preplay_d[3][:time])
        else:
            state_all = np.array(self.preplay_d[3])
        max_float = np.finfo(dtype=float).max

        def similar_func(s):
            distance = np.linalg.norm(s - state)
            if distance < epsilon :
                return max_float
            else:
                return distance
        similar_func_v = np.vectorize(similar_func)

        similar_list = similar_func_v(state_all)
        similar_state_index = np.argmin(similar_list)
        return similar_state_index

    def _similar_state_index_norm_gpu(self, state, time):
        epsilon = 0.000001
        state_all = []
        length = 0
        if time < self.preplay_data_size:  # during the first sweep of the History Data
            state_all = self.preplay_d[3][:time]
            length = time
        else:
            state_all = self.preplay_d[3]
            length = self.preplay_data_size

        max_float = np.finfo(dtype=float).max
        state_repeat = cuda.cupy.array(list(state.reshape(-1))*length)
        state_all = cuda.to_gpu(state_all.copy()).reshape(-1)

        power_list = cuda.cupy.sqrt(cuda.cupy.sum(cuda.cupy.power(state_repeat - state_all, 2).reshape(length, len(state_all)/length), axis=1))

        while True:
            similar_state_index = int((cuda.cupy.argmin(power_list)))
            if power_list[similar_state_index] > epsilon:
                break
            else:
                power_list[similar_state_index]  = max_float

        return similar_state_index

    def _similar_state_index_cos(self, state, time):
        epsilon = 0.000001
        state_all = []
        if time < self.preplay_data_size:  # during the first sweep of the History Data
            state_all = self.preplay_d[3][:time]
        else:
            state_all = self.preplay_d[3]

        while np.sum(state) == 0:
            preplay_index = 0
            if time < self.data_size:  # during the first sweep of the History Data
                preplay_index = np.random.randint(0, time)
            else:
                preplay_index = np.random.randint(0, self.data_size)
            state = state_all[preplay_index]
        similar_func = lambda s : 100 if np.sum(s) == 0 else (cosine(state, s) if cosine(state, s) > epsilon else 100.0)
        similar_list = map(similar_func, state_all)
        similar_state_index = np.argmin(similar_list)
        return similar_state_index

    def _similar_state_index_cos_gpu(self, state, time):
        epsilon = 0.000001
        max_float = np.finfo(dtype=float).max
        state_all = []
        if time < self.preplay_data_size:  # during the first sweep of the History Data
            state_all = self.preplay_d[3][:time]
        else:
            state_all = self.preplay_d[3]

        while np.sum(state) == 0:
            preplay_index = 0
            if time < self.data_size:  # during the first sweep of the History Data
                preplay_index = np.random.randint(0, time)
            else:
                preplay_index = np.random.randint(0, self.data_size)
            state = state_all[preplay_index]
        state = cuda.to_gpu(state)
        state_all = cuda.to_gpu(state_all)
        def similar_func(s):
            if cuda.cupy.abs(s).sum() < epsilon:
                return 100
            else:
                cos_similar = self._cosine_gpu(s, state)
                if cos_similar < epsilon:
                    return 100
                else:
                    return float(cos_similar)
        similar_list = map(similar_func, state_all)
        similar_state_index = int(np.argmin(similar_list))
        return similar_state_index

    def _cosine_gpu(self, s1, s2):
        s1 = s1.reshape(-1)
        s2 = s2.reshape(-1)
        return 1 - cuda.cupy.dot(s1, s2)/(self._norm_gpu(s1, s2) * self._norm_gpu(s1, s2))

    def _norm_gpu(self, s1, s2):
        return cuda.cupy.sqrt((s1-s2).sum())


    def end_episode(self, time, last_state, action, reward):
        self.stock(time, last_state, action, reward, last_state, True)
        replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay = \
            self.replay(time)

        return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay
