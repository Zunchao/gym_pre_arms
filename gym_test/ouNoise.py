import numpy as np
# Exploration noise generator which uses Ornstein-Uhlenbeck process.
class OUNoise:
    def __init__(self, num_actions, mu=0, theta=0.15, sigma=0.2, delta=0.5):
        self.num_actions = num_actions
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.delta = delta

        self.reset()

    def generate(self):
        #x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        #self.state = x + dx
        #return self.state
        '''
        prev_ou_level = self.ou_level
        drift = self.theta * (self.mu - prev_ou_level) * self.delta
        randomness = np.random.normal(loc=0, scale=np.sqrt(self.delta)*self.sigma, size=None) # Brownian motion
        self.ou_level = prev_ou_level + drift + randomness
        print('ou',self.ou_level)
        '''

        drift = self.theta * (self.mu - self.prev_ou) * self.delta
        randomness = np.random.normal(loc=0, scale=np.sqrt(self.delta) * self.sigma, size=None)  # Brownian motion
        ou_level = self.prev_ou + drift + randomness
        self.prev_ou = ou_level
        return ou_level

    def reset(self):
        #self.ou_level = np.ones(self.num_actions) * self.mu
        self.prev_ou = np.ones(self.num_actions) * self.mu
'''
if __name__ == "__main__":
    noise = OUNoise(3)
    outputs = []
    for _ in range(1000):
        outputs.append(noise.generate())
    import matplotlib.pyplot as plt
    plt.plot(outputs)
    plt.show()
'''