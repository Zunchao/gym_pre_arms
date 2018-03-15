from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

class SimEnv3Joints():
    '''build a simple 2d env for 3 joint arm moving in a limited space, basic functions from https://github.com/msieb1/policy-search-for-2d-arm'''
    def __init__(self):
        # env param
        self.numJoints = 3
        self.lengthJoint = 10
        self.dofArm = 3
        #self.workspaceLength = 100
        #self.workspaceWidth = 100
        # robot state: theta1 theta1' theta2 theta2' theta3 theta3'
        self.initState = np.array([pi/2, 0., 0., 0., 0., 0.])
        # target position: Xt Yt
        self.setTargetPosi = np.array([10., 10.])
        self.numBasicFun = 5
        self.numTrajSteps = 20
        self.dt = 0.01
        self.kp = 1400.
        self.masses = np.ones(self.numJoints)
        # em param
        self.lamta = 7
        self.numDim = 15
        self.numSamples = 50
        self.maxIter = 200
        self.numTrials = 10

        self.pjoint_ = np.zeros((self.dofArm + 1, 2))
        self.px = np.zeros((1, 4))
        self.py = np.zeros((1, 4))
        self.fig = plt.figure()

        # global pjoint_global

    def transJoint(self, thetaSingle):
        Tp = np.matrix(((cos(thetaSingle), sin(thetaSingle), self.lengthJoint*sin(thetaSingle)), \
                           (-sin(thetaSingle), cos(thetaSingle), self.lengthJoint*cos(thetaSingle)), \
                           (0, 0, 1)))
        return Tp

    def fKinematics(self, theta_):
        T = np.eye(self.dofArm)
        for i in range(self.dofArm):
            T = T*self.transJoint(theta_[i])
            self.pjoint_[i + 1, :] = [T[0, -1], T[1, -1]]
        pT = [T[0,-1],T[1,-1]]
        return pT

    def pjoint_global_update(self):

        # print('ahsdui', pjoint_[3,1])
        for i in range(4):
            self.px[0][i] = self.pjoint_[i, 0]
            self.py[0][i] = self.pjoint_[i, 1]
        #print('px', self.px, 'py', self.py)
        # plt.cla()
        # ax.plot(self.px, self.py)
        # plt.pause(1)
        # self.fig.canvas.draw()

    def pjoint_global(self):
        global pjoint_
        pjoint_ = self.pjoint_
        # pjoint_ = self.pjoint_global_update(pjoint_)
        return pjoint_

    def jointPositions(self, theta_):
        T = np.eye(self.dofArm)
        each_pjoint_ = self.pjoint_
        for i in range(self.dofArm):
            T = T*self.transJoint(theta_[i])
            #print('matriac: ',[T[0, -1], T[1, -1]])
            self.pjoint_[i + 1, :] = [T[0, -1], T[1, -1]]
        # print('position fo each joint : ', pjoint_[:, 0], pjoint_[:, 1])
        # self.pjoint_ = pjoint_
        #return each_pjoint_

    def jacobianMatrix(self,theta_):
        Jmatrix_ = np.matrix((( - sin(sum(theta_[0:1])) - sin(theta_[0]) - sin(sum(theta_[:])), - sin(sum(theta_[0:1])) - sin(sum(theta_[:])), -sin(sum(theta_[:]))), \
                              (    cos(sum(theta_[0:1])) + cos(theta_[0]) + cos(sum(theta_[:])),   cos(sum(theta_[0:1])) + cos(sum(theta_[:])),  cos(sum(theta_[:]))), \
                              (0, 0, 0), \
                              (0, 0, 0), \
                              (0, 0, 0), \
                              (1, 1, 1)))
        return Jmatrix_

    def transitionFunction(self, x, action):
        xnew = np.zeros(x.shape)
        xnew[1::2] = x[1::2] + (action / self.masses) * self.dt
        xnew[::2] = x[::2] + xnew[1::2] * self.dt
        return xnew

    def calRewardFunction(self,theta_):
        # c1 = -1e-4
        # c2 = -1e-4
        # uf = u*u.T
        # df = distance(target, ee position)
        # r = c1*df + c2*uf
        q_desired = self.policyRepresentationDMP(np.reshape(theta_, (-1, self.numBasicFun)).T)
        sim_traj = self.simEnvbasic(q_desired)

        uf = -1e-4
        uCost = uf * np.linalg.norm(sim_traj[1]) ** 2

        df = -1e4
        dis_ee = self.setTargetPosi - sim_traj[2][-1, :]
        rCost = np.dot(dis_ee, dis_ee) * df

        return uCost + rCost, sim_traj[0]

    def policyRepresentationDMP(self, dmp_w):
        alphaz = 3.3
        alpha = 25.
        beta = 0.8
        tau = 1.
        Ts = 1.
        g = self.initState[::2]  # goal?

        C = np.exp(-alphaz*np.arange(self.numBasicFun)/(self.numBasicFun-1)*Ts)
        H = 0.5 / (0.65 * np.diff(C)**2)
        H = np.append(H,H[-1])

        q = np.zeros((self.numTrajSteps, 2*self.numJoints))
        q[0,:] = self.initState
        # phase variable z
        x = np.ones(self.numTrajSteps)

        for i in range(self.numTrajSteps-1):
            psi = np.exp(-H*(x[i]-C)**2)
            f = np.dot(dmp_w.T, psi) * x[i] / np.sum(psi)
            qdd_des = (alpha * (beta * ( g - q[i,::2] ) - ( q[i,1::2] / tau ) ) + f.T ) * tau**2
            q[i+1,1::2] = q[i,1::2] + qdd_des * self.dt  # theta dot?
            q[i+1,::2] = q[i,::2] + q[i+1,1::2] * self.dt
            xd = -alphaz*x[i]*tau
            x[i+1] = x[i] + xd * self.dt
        return q

    def simEnvbasic(self, q_desired):
        mat_state_ = np.zeros((self.numJoints, 2*self.numJoints))
        mat_state_[:, ::2] = self.kp * np.eye(self.numJoints)
        mat_state_[:, 1::2] = 2 * np.sqrt(self.kp) * np.eye(self.numJoints)

        q = np.zeros((q_desired.shape[0], 2 * self.numJoints))
        q[0, :] = self.initState

        # global pjoint_
        # pjoint_ = np.zeros((self.dofArm+1, 2))
        self.pjoint_global()

        pos = np.zeros((q_desired.shape[0], 2))
        vel = np.zeros((q_desired.shape[0], 2))
        pos[0, :] = self.fKinematics(q[0, ::2])

        u = np.zeros((q_desired.shape[0], self.numJoints))

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

    # maximum likelihood
    def calculate_reward_and_theta(self, Mu_w, Sigma_w):
        numDim = self.numDim
        numSamples = self.numSamples
        traj = np.zeros((self.numTrajSteps, self.numJoints))
        theta = np.zeros((numDim, numSamples))
        R = np.zeros(numSamples)
        for i in range(0, numSamples):
            sample = np.random.multivariate_normal(Mu_w, Sigma_w)
            reward, simtraj = self.calRewardFunction(sample)
            theta[:,i] = sample
            R[i] = reward
            traj = simtraj
        return R, theta, traj

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

    def run(self):
        maxIter = self.maxIter
        numDim = self.numDim
        numSamples = self.numSamples
        numTrials = self.numTrials

        R_mean_storage = np.zeros((maxIter, numTrials))
        R_mean = np.zeros(maxIter)
        R_std = np.zeros(maxIter)

        # self.pjoint_global()
        #self.movie_init()
        #self.movie_animate(0)
        #ani = animation.FuncAnimation(self.fig, self.movie_animate, frames=300,
        #                              interval=100, blit=True, init_func=self.movie_init)
        #plt.show()
        #fig = plt.figure()
        #ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
        #                     xlim=(-30, 30), ylim=(-30, 30))
        #ax.grid()

        #pos = np.zeros((self.numSamples, 2*self.numJoints))

        for t in range(0, numTrials):
            print('trials No. : ', t)
            R_old = np.zeros(numSamples)
            Mu_w = np.zeros(numDim)
            Sigma_w = np.eye(numDim) * 1e6
            for k in range(0, maxIter):
                R, theta, traj = self.calculate_reward_and_theta(Mu_w, Sigma_w)
                # plot end config of sampled trajectories
                self.pjoint_global_update()
                plt.axis(([-30, 30, -30, 30]))
                plt.grid()
                plt.ion()
                plt.plot(self.px[0], self.py[0],'b.-')
                plt.plot(self.setTargetPosi[0], self.setTargetPosi[1], 'ro')
                plt.pause(0.00001)
                plt.cla()

                if np.linalg.norm(np.mean(R_old) - np.mean(R)) < 1e-3:
                    break
                w = self.calculate_w(R, theta)
                Mu_w, Sigma_w = self.update_omega(w, theta)
                Sigma_w += np.eye(numDim)
                mR = np.mean(R)
                R_mean_storage[k, t] = mR
                R_old = R
                if k == maxIter and t == numTrials:
                    print(np.mean(R))
            print('start trajactory of trial ', t)
            # plot trajectory of last iteration
            for j in range(traj.shape[0]-1):
                self.jointPositions(traj[j + 1,::2])
                self.pjoint_global_update()
                #pos[i + 1, :] = self.fKinematics(q[i + 1, ::2])
                plt.axis(([-30, 30, -30, 30]))
                plt.grid()
                plt.ion()
                plt.plot(self.px[0], self.py[0], 'k.-')
                plt.plot(self.setTargetPosi[0], self.setTargetPosi[1], 'ro')
                plt.pause(0.000001)
                plt.cla()

        R_mean = np.mean(R_mean_storage, axis=1)
        R_std = np.sqrt(np.diag(np.cov(R_mean_storage)))
        print("Average return of final policy: ")
        print(R_mean[-1])
        print("\n")

    # try to use animate() to visualise but failed
    # the following 2 functions useless
    def movie_init(self):
        global line
        print('movie init')

        ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-30, 30), ylim=(-30, 30))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        line.set_data([], [])
        return line,

    def movie_animate(self, i):
        print('move animate')
        line.set_data(self.px, self.py)
        return line,


#if __name__ == '__main__':
#global test, fig

test = SimEnv3Joints()

# test.movie_init()
# test.pjoint_global()
# test.pjoint_global_update()
# test.movie_animate(0)

# ani.save('test_new.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

test.run()
plt.close(test.fig)

'''
global test
test = SimEnv3Joints()

test.pjoint_global()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-30, 30), ylim=(-30, 30))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)

def movie_init():
    #global line, fig
    line.set_data([], [])
    return line,

def movie_animate(i):
    # print('sdgufz')
    test.pjoint_global()
    # print('jasnh', pjoint_[:, 0], pjoint_[:, 1])
    line.set_data(*test.pjoint_global_update(pjoint_))
    return line,

movie_animate(0)

ani = animation.FuncAnimation(fig, movie_animate, frames=300,
                              interval=100, blit=True, init_func=movie_init)
# ani.save('test_new.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
'''