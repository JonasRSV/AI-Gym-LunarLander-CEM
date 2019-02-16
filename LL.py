import gym
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance


def relu(X):
    return np.maximum(0, X)


class CEM():
    def __init__(self, state_space, hidden_space, action_space):
        self.state_space = state_space
        self.hidden_space = hidden_space
        self.action_space = action_space
        self.covariances = (
            np.eye((state_space + 1) * (hidden_space + 1) +
                   (hidden_space + 1) * action_space) - 0.5) * 2
        self.means = (
            np.random.rand((state_space + 1) * (hidden_space + 1) +
                           (hidden_space + 1) * action_space) - 0.5) * 2

        self.best_transform = None
        self.mean_transform = None
        self.best_score = -100000

    def generate(self, x):
        self.agents = []
        self.scores = []
        for i in range(x):
            parameter_space = np.random.multivariate_normal(
                self.means, self.covariances)
            in_transform = parameter_space[:(self.state_space + 1) * (
                self.hidden_space + 1)]
            out_transform = parameter_space[(self.state_space + 1) * (
                self.hidden_space + 1):]

            self.agents.append([
                in_transform.reshape(-1, self.state_space + 1),
                out_transform.reshape(-1, self.action_space)
            ])

            self.scores.append([i, 0])

        in_transform = self.means[:(self.state_space + 1) * (
            self.hidden_space + 1)]
        out_transform = self.means[(self.state_space + 1) * (
            self.hidden_space + 1):]
        self.mean_transform = [
            in_transform.reshape(-1, self.state_space + 1),
            out_transform.reshape(-1, self.action_space)
        ]

    def update(self, apex):
        self.scores.sort(key=lambda x: x[1], reverse=True)
        samples = []
        for idx, _ in self.scores[:int(len(self.scores) * apex)]:
            in_transform = self.agents[idx][0].reshape(-1)
            out_transform = self.agents[idx][1].reshape(-1)
            samples.append(np.concatenate([in_transform, out_transform]))

        cov = EmpiricalCovariance(assume_centered=False).fit(samples)
        self.means = cov.location_
        self.covariances = cov.covariance_

    def act(self, a, s, discrete=True):
        s = np.append(s, [1])
        if discrete:
            return np.argmax(relu(self.agents[a][0] @ s) @ self.agents[a][1])
        else:
            return relu(self.agents[a][0] @ s) @ self.agents[a][1]

    def score(self, a, s):
        self.scores[a][1] = s

    def save(self, a, score):
        if score > self.best_score:
            self.best_score = score
            self.best_transform = self.agents[a]

    def act_best(self, s, discrete=True):
        s = np.append(s, [1])
        if discrete:
            return np.argmax(
                relu(self.best_transform[0] @ s) @ self.best_transform[1])
        else:
            return relu(self.best_transform[0] @ s) @ self.best_transform[1]

    def act_mean(self, s, discrete=True):
        s = np.append(s, [1])
        if discrete:
            return np.argmax(
                relu(self.mean_transform[0] @ s) @ self.mean_transform[1])
        else:
            return relu(self.mean_transform[0] @ s) @ self.mean_transform[1]


def play_best(ll, cem, discrete):
    while True:
        s = ll.reset()
        done = False
        while not done:
            ll.render()
            a = cem.act_best(s, discrete)
            print(a)
            s, r, done, _ = ll.step(a)

    sys.exit(0)


def play_mean(ll, cem, discrete):
    while True:
        s = ll.reset()
        done = False
        while not done:
            ll.render()
            s, r, done, _ = ll.step(cem.act_mean(s, discrete))

    sys.exit(0)


if __name__ == "__main__":
    ll = gym.make("LunarLander-v2")
    DISCRETE = True
    cem = None
    if len(sys.argv) > 1:
        with open("model.pkl", "rb") as model:
            cem = pickle.load(model)
    else:
        cem = CEM(8, 9, 4)

    if len(sys.argv) > 1 and sys.argv[1] == "-p":
        play_best(ll, cem, DISCRETE)

    if len(sys.argv) > 1 and sys.argv[1] == "-m":
        play_mean(ll, cem, DISCRETE)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    AGENTS = 1000
    ITER_PER_AGENT = 2
    generations = []
    best_scores = []
    average_scores = []
    mean_scores = []
    generation = 0
    fig.tight_layout()
    ax1.set_title("best")
    ax2.set_title("averages")
    ax3.set_title("mean")
    try:
        while True:
            cem.generate(AGENTS)
            best_score = 0
            average_score = 0
            """ Evaluate Agents. """
            for agent in range(AGENTS):
                cumulative_score = 0
                for _ in range(ITER_PER_AGENT):
                    s = ll.reset()
                    done = False
                    while not done:
                        # ll.render()
                        s, r, done, _ = ll.step(
                            cem.act(agent, s, discrete=DISCRETE))
                        cumulative_score += r
                cem.score(agent, cumulative_score // ITER_PER_AGENT)
                best_score = max(best_score,
                                 cumulative_score // ITER_PER_AGENT)
                average_score += cumulative_score // ITER_PER_AGENT
                cem.save(agent, cumulative_score // ITER_PER_AGENT)
            """ Evaluate Mean. """

            cumulative_score = 0
            for _ in range(ITER_PER_AGENT):
                s = ll.reset()
                done = False
                while not done:
                    # ll.render()
                    s, r, done, _ = ll.step(cem.act_mean(s, DISCRETE))
                    cumulative_score += r

            mean_scores.append(cumulative_score // ITER_PER_AGENT)
            generations.append(generation)
            best_scores.append(best_score)
            average_scores.append(average_score / AGENTS)
            generation += 1
            ax1.plot(generations, best_scores)
            ax2.plot(generations, average_scores)
            ax3.plot(generations, mean_scores)
            cem.update(0.1)
            plt.pause(0.001)
    except KeyboardInterrupt:
        with open("model.pkl", "wb") as model:
            pickle.dump(cem, model)
