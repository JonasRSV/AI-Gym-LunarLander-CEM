import gym
import numpy as np

ll = gym.make("CartPole-v0")

MUST_PASS = 2
iters = 0

while True:
    agent = np.random.rand(2, 4)

    cumul_rew = 0
    for _ in range(MUST_PASS):
        s = ll.reset()
        done = False
        while not done:
            # ll.render()
            s, r, done, _ = ll.step(np.argmax(agent @ s))
            cumul_rew += r

    if cumul_rew >= 200 * MUST_PASS:
        print("DONE IN {}".format(iters))
        while True:
            for _ in range(MUST_PASS):
                s = ll.reset()
                done = False
                while not done:
                    ll.render()
                    s, r, done, _ = ll.step(np.argmax(agent @ s))

    iters += 1
