#encoding:utf8
import pinyin
import numpy as np
import copy
class HmmPinyinModel(object):
    def __init__(self, states, vocab, transition=None, emission=None):
        """
        :param states: list HMM hidden states  
        :param transition: transition probability matrix
        :param emission: emission probability
        :param vocab
        :param pi: initial probability
        """

        if not isinstance(states, list) or not isinstance(vocab, list):
            raise Exception('states or vocab is not a list')

        _start_state = '<S>'
        _end_state = '<E>'
        self.states = [_start_state] + states + [_end_state]
        self.N = len(self.states)
        self.transition = np.zeros((len(self.states), len(self.states)), dtype=np.float64)
        self.emssion = np.zeros((len(self.states), len(vocab)), dtype=np.float64)
        self.vocab = vocab
        self.T = len(self.vocab)
        self.state2ix = {}
        self.ix2state={}
        for ix, st in enumerate(self.states):
            self.state2ix[st] = ix
            self.ix2state[ix] = st
        self.obs2ix={}
        for i, obs in enumerate(self.vocab):
            self.obs2ix[obs] = i

        self.oov=[]


        for i in range(self.N):
            for j in range(self.N):
                self.transition[i][j] = transition[i][j]
        for i in range(self.N):
            for j in range(self.T):
                self.emssion[i][j] = emission[i][j]

    def _transition_prob(self, frm, to, prob):

        assert(0 <= frm and frm < self.N)
        assert(0 <= to and to < self.N)
        assert(0.0 <= prob and prob <= 1.0)
        self.transition[frm][to] = prob

    def _emission_prob(self, state, obs, prob):
        assert (state >= 0 and state < self.N)
        assert (obs >= 0 and obs < self.T)
        assert(0.0 <= prob and prob <= 1.0)
        self.emssion[state][obs] = prob

    def _sent_pyin(self,sent):
        """
        return the pinyin of sent
        :param sent: 
        :return: 
        """
        pyins = pinyin.get(sent, delimiter= ' ', format='strip')
        return  pyins

    def train(self, dataset, smooth=None):
        """
        train hmm model with data one iterate
        :param data: train data 
        :param smooth: smooth method
        :return: 
        """
        #count number of transition from a to b
        with open(dataset) as df:
            for sent in df:
                sent = sent.strip()
                for i in range(len(sent)-1):
                    frm = sent[i]
                    to = sent[i+1]
                    try:
                        ix_frm = self.state2ix[frm]
                        ix_to = self.state2ix[to]
                        self.transition[ix_frm][ix_to] += 1.0
                    except Exception as ex:
                        print('exception: {}'.format(ex))
                        continue
                pinyins = self._sent_pyin(sent)
                pinyins = pinyins.strip().split()
                for hanzi, pyin in zip(list(sent), pinyins):
                    try:
                        ix_ob = self.obs2ix[pyin]
                        ix_st = self.state2ix[hanzi]
                        self.emssion[ix_st][ix_ob] += 1.0
                    except Exception as ex:
                        print('exception: {}'.format(ex))
                        continue


        #scale to probs
        sums = np.sum(self.transition, axis=1)
        sums[sums<=0.0] = 1.0
        self.transition = self.transition / sums[:,None]

        sums = np.sum(self.emssion, axis=1)
        sums[sums<=0.0] = 1.0
        self.emssion = self.emssion /sums[:, None]

    def decode_viterbi(self, pyin_sent):
        pyins = pyin_sent.split()
        T = len(pyins)
        tmp = [None]*T

        viterbi = [tmp]*self.N
        backpointer = copy.deepcopy(viterbi)
        def max_v(st, obst):
            cur_max = -1000.0
            cur_s = -1
            ixx = -1
            for sst in range(1, self.N):
                tmp = viterbi[sst][obst-1]*self.transition[sst][st]*self.emssion[st][obst]
                bck =  viterbi[sst][obst-1]*self.transition[sst][st]
                if tmp > cur_max:
                    cur_max = tmp
                if cur_s < bck:
                    cur_s = bck
                    ixx = sst
            return  cur_max, ixx

        for s in range(1,self.N):
            viterbi[s][1] = self.transition[0][s]*self.emssion[s][1]
            backpointer[s][1] = 0

        for t in range(2, T):
            for st in range(1, self.N):
                max_viterbi, ixx = max_v(st, t)
                viterbi[st][t] = max_viterbi
                backpointer[st][t] = ixx

        v, i = max_v(self.N-1, T-1)
        viterbi[self.N-1][T-1] = v
        backpointer[self.N-1][T-1] = i

        states_list = []
        s = backpointer[self.N-1][T-1]
        obs = T-1
        while obs >= 0:
            states_list.append(s)
            obs -= 1
            s = backpointer[s][obs]
        states_list.reverse()
        return states_list
    def pyin2hanzi(self, pyin_sent):
        states = self.decode_viterbi(pyin_sent)
        s = []
        for i in states:
            s.append(self.ix2state[i])

        return ''.join(s)

    def save(self):
        np.savetxt("model/trans.txt", self.transition, delimiter=' ')
        np.savetxt("model/emission.txt", self.emssion, delimiter=' ')


states = []
vocab = []

with open('data/pinyin2hanzi.txt') as inf:
    for l in inf:
        l = l.strip()
        l = l.split()
        vocab.append(l[0])
        states.extend(list(l[1]))
with open('model/states.txt', 'w') as sf, open('model/vocab', 'w') as wf:
    sf.write(' '.join(states))
    wf.write(' '.join(vocab))

trans = np.loadtxt('model/trans.txt', dtype=np.float64, delimiter=' ')
emiss = np.loadtxt('model/emission.txt', dtype=np.float64,  delimiter=' ')

hmmpyin = HmmPinyinModel(states, vocab, transition=trans, emission=emiss)

#hmmpyin.train('data/data.txt')
#hmmpyin.save()
print("load model success")

pyin = "wo men de gong si"
print(hmmpyin.pyin2hanzi(pyin))