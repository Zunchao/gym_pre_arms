from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
import gym
from scipy.stats import multivariate_normal

class SimEnv3Joints():
    '''build a simple 2d env for 3 joint arm moving in a limited space, basic functions from https://github.com/msieb1/policy-search-for-2d-arm'''
    def __init__(self):
        self.numTrajSteps = 20
        # em param
        self.lamta = 7
        self.numActionDim = 4
        self.numSamples = 30

        self.numDimOfSample = 35
        # self.numDimOfSample = self.dofArm*self.numBasicFun
        self.maxIter = 100
        self.numTrials = 10
        self.env = gym.make('FetchReach-v0')
        self.dofArm = 7
        self.numBasicFun = 5
        self.dt = 0.005
        self.kp = 1400.


    def policyRepresentationDMP(self, dmp_w):
        # param set up by Table 1 in Schaal's paper
        # Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors
        alpha_z = 25.
        beta_z = alpha_z/4
        alpha_x = alpha_z/3
        tau = 1.
        Ts = 1.
        g = init_positions

        C = np.exp(-alpha_x*np.arange(self.numBasicFun)/(self.numBasicFun-1)*Ts)
        H = 0.5 / (0.65 * np.diff(C)**2)
        H = np.append(H,H[-1])

        qp = np.zeros((self.numTrajSteps, self.dofArm))
        qp[0, :] = init_positions
        qv = np.zeros((self.numTrajSteps, self.dofArm))
        qv[0, :] = init_velocities
        # phase variable z
        x = np.ones(self.numTrajSteps)

        for i in range(self.numTrajSteps-1):
            psi = np.exp(-H*(x[i]-C)**2)
            f = np.dot(dmp_w.T, psi) * x[i] / np.sum(psi)
            # print('dmp_w', dmp_w, f)
            qa = (alpha_z * (beta_z * ( g - qp[i,:] ) - ( qv[i,:] / tau ) ) + f.T ) * tau**2
            qv[i+1,:] = qv[i,:] + qa * self.dt  # theta dot?
            qp[i+1,:] = qp[i,:] + qv[i+1,:] * self.dt
            xd = -alpha_x*x[i]*tau
            x[i+1] = x[i] + xd * self.dt
        return qp, qv

    def simEnvbasic(self, q_desired):
        mat_state_ = np.zeros((self.dofArm, 2*self.dofArm))
        mat_state_[:, ::2] = self.kp * np.eye(self.dofArm)
        mat_state_[:, 1::2] = 2 * np.sqrt(self.kp) * np.eye(self.dofArm)

        q = np.zeros((q_desired.shape[0], 2 * self.dofArm))
        q[0, :] = self.initState

        # global pjoint_
        # pjoint_ = np.zeros((self.dofArm+1, 2))

        pos = np.zeros((q_desired.shape[0], 2))
        vel = np.zeros((q_desired.shape[0], 2))
        pos[0, :] = self.fKinematics(q[0, ::2])

        u = np.zeros((q_desired.shape[0], self.dofArm))

        for i in range(q_desired.shape[0] - 1):
            u[i, :] = np.dot(mat_state_, (q_desired[i, :] - q[i, :]).T)
            q[i + 1, :] = self.transitionFunction(q[i, :], u[i, :])
            pos[i + 1, :] = self.fKinematics(q[i + 1, ::2])
            # pjoint_ = self.jointPositions(q[-1, ::2])
            # print('pjoint : ', pjoint_)
            # px, py = self.pjoint_global_update()
            # self.movie_animate(0)
            vel[i + 1, :] = np.dot(self.jacobianMatrix(q[i + 1, ::2])[:2, :], q[i + 1, 1::2].T)  # linear velocity

        #plt.plot(pjoint[:,0],pjoint[:,1],'r.-')
        #plt.show()
        return [q, u, pos, vel]

    def calRewardFunction(self, action):
        env = self.env
        env.render()
        observation, reward, done, info = env.step(action)
        #print('reward1', reward)
        #reward == env.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)
        #print('reward2', reward)
        # if done:
            # break
        return reward

    # maximum likelihood
    def calculate_wsample(self, Mu_w, Sigma_w):
        numDimOfSample = self.numDimOfSample
        numSamples = self.numSamples
        theta = np.zeros((numDimOfSample, numSamples))
        R = np.zeros(numSamples)
        action = np.zeros(numDimOfSample)
        for i in range(0, numSamples):
            samples = np.random.multivariate_normal(Mu_w, Sigma_w)
            #dsamples = multivariate_normal.pdf(samples, Mu_w, Sigma_w)
            #csamples = np.random.choice([-1, 1], 3)
            #dsamples = dsamples * csamples
            #action[:] = np.sin(samples)
            #action[:3] = dsamples
            #action = np.zeros(numActionDim)
            theta[:,i] = action
            #print('action', action, Mu_w, Sigma_w)
            reward = self.calRewardFunction(action)
            R[i] = reward
        return R, theta

    def calculate_w(self, R, theta):
        # calculate the weights by success probability
        lamta = self.lamta
        numSamples = self.numSamples
        w = np.zeros(numSamples)
        beta = lamta / (np.max(R) - np.min(R))
        for i in range(0, numSamples):
            w[i] = np.exp(beta * (R[i] - np.max(R)))
        return w

    def update_omega(self, w, theta):
        Mu_w = 0
        for l in range(0, self.numSamples):
            Mu_w += w[l]*theta[:,l]
        Mu_w /= np.sum(w)

        Sigma_w = 0
        for l in range(0, self.numSamples):
            Sigma_w += w[l]*(np.outer((theta[:,l] - Mu_w),(theta[:,l] - Mu_w)))
        Sigma_w /= np.sum(w)

        return Mu_w, Sigma_w

    def set_global_params(self):
        global init_positions
        global init_velocities
        global target_position

        init_positions = 0
        init_velocities = 0
        target_position = 0


    def run(self):
        maxIter = self.maxIter
        numActionDim = self.numActionDim
        numSamples = self.numSamples
        numTrials = self.numTrials
        self.set_global_params()

        R_mean_storage = np.zeros((maxIter, numTrials))
        R_mean = np.zeros(maxIter)
        R_std = np.zeros(maxIter)

        env = self.env
        env.render()
        observation = env.reset()

        action = np.zeros(numActionDim)
        #observation, reward, done, info = env.step(action)

        for t in range(0, numTrials):
            print('trials No. : ', t)
            observation = env.reset()
            init_positions = observation['joint_positions']
            init_velocities = observation['joint_velocities']
            target_position = observation['desired_goal']

            R_old = np.zeros(numSamples)
            Mu_w = np.zeros(numActionDim)
            Sigma_w = np.eye(numActionDim)*1e6

            for k in range(0, maxIter):
                R, theta = self.calculate_wsample(Mu_w, Sigma_w)
                #print('r theta', R, theta)

                if np.linalg.norm(np.mean(R_old) - np.mean(R)) < 1e-3:
                    print('iter', k)
                    break
                w = self.calculate_w(R, theta)
                Mu_w, Sigma_w = self.update_omega(w, theta)
                Sigma_w += np.eye(numActionDim)
                mR = np.mean(R)
                R_mean_storage[k, t] = mR
                R_old = R
                if k == maxIter and t == numTrials:
                    print(np.mean(R))

        R_mean = np.mean(R_mean_storage, axis=1)
        R_std = np.sqrt(np.diag(np.cov(R_mean_storage)))
        print("Average return of final policy: ")
        print(R_mean[-1])
        print("\n")

# main function
if __name__ == '__main__':
    test = SimEnv3Joints()
    test.run()
    #plt.close(test.fig)
