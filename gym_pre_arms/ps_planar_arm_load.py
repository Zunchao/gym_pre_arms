from math import pi, sin, cos, sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GPy
from IPython.display import display
import time
from csv_writer_reader import CSV_Writer_Reader
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

class SimEnv3Joints():
    '''build a simple 2d env for 3 joint arm moving in a limited space, basic functions from https://github.com/msieb1/policy-search-for-2d-arm'''
    def __init__(self):
        # env param
        self.numJoints = 3
        self.lengthJoint = 10
        self.dofArm = 3
        # self.workspaceLength = 100
        # self.workspaceWidth = 100
        # robot state: theta1 theta1' theta2 theta2' theta3 theta3'
        self.initState = np.array([pi/2, 0., 0., 0., 0., 0.])
        # target position: Xt Yt
        self.setTargetPosi = np.array([10., 10.])
        self.numBasicFun = 5
        self.numTrajSteps = 20
        self.dt = 0.005
        self.kp = 1400.
        self.masses = np.ones(self.numJoints)
        # em param
        self.lamta = 7
        self.numDimOfSample = 15
        # self.numDimOfSample = self.dofArm*self.numBasicFun
        self.numSamples = 30
        self.maxIter = 1000
        self.numTrials = 10
        self.pjoint_ = np.zeros((self.dofArm + 1, 2))
        self.px = np.zeros((1, 4))
        self.py = np.zeros((1, 4))
        self.fig = plt.figure()
        self.csvwr = CSV_Writer_Reader()


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
        # transform matrix to vectors x and y for the plot
        for i in range(4):
            self.px[0][i] = self.pjoint_[i, 0]
            self.py[0][i] = self.pjoint_[i, 1]

    def pjoint_global(self):
        # set global param
        global pjoint_
        pjoint_ = self.pjoint_
        # pjoint_ = self.pjoint_global_update(pjoint_)
        return pjoint_

    def jointPositions(self, theta_):
        # calculate positions of each joint
        T = np.eye(self.dofArm)
        each_pjoint_ = self.pjoint_
        for i in range(self.dofArm):
            T = T*self.transJoint(theta_[i])
            #print('matriac: ',[T[0, -1], T[1, -1]])
            self.pjoint_[i + 1, :] = [T[0, -1], T[1, -1]]

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

    def calRewardFunction(self,theta_, settarget):
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
        dis_ee = settarget - sim_traj[2][-1, :]
        rCost = np.dot(dis_ee, dis_ee) * df

        return uCost + rCost, sim_traj[0]

    def policyRepresentationDMP(self, dmp_w):
        # param set up by Table 1 in Schaal's paper
        # Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors
        alpha_z = 25.
        beta_z = alpha_z/4
        alpha_x = alpha_z/3
        tau = 1.
        Ts = 1.
        g = self.initState[::2]  # goal?

        C = np.exp(-alpha_x*np.arange(self.numBasicFun)/(self.numBasicFun-1)*Ts)
        H = 0.5 / (0.65 * np.diff(C)**2)
        H = np.append(H,H[-1])

        q = np.zeros((self.numTrajSteps, 2*self.numJoints))
        q[0,:] = self.initState
        # phase variable z
        x = np.ones(self.numTrajSteps)

        for i in range(self.numTrajSteps-1):
            psi = np.exp(-H*(x[i]-C)**2)
            f = np.dot(dmp_w.T, psi) * x[i] / np.sum(psi)
            # print('dmp_w', dmp_w, f)
            qdd_des = (alpha_z * (beta_z * ( g - q[i,::2] ) - ( q[i,1::2] / tau ) ) + f.T ) * tau**2
            q[i+1,1::2] = q[i,1::2] + qdd_des * self.dt  # theta dot?
            q[i+1,::2] = q[i,::2] + q[i+1,1::2] * self.dt
            xd = -alpha_x*x[i]*tau
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
    def calculate_reward_and_theta(self, Mu_w, Sigma_w, settarget):
        numDimOfSample = self.numDimOfSample
        numSamples = self.numSamples
        traj = np.zeros((self.numTrajSteps, self.numJoints))
        theta = np.zeros((numDimOfSample, numSamples))
        R = np.zeros(numSamples)
        for i in range(0, numSamples):
            # x = np.linspace(-1, 1, numDimOfSample)
            sample = np.random.multivariate_normal(Mu_w, Sigma_w)
            # print('sample', Mu_w, sample)
            reward, simtraj = self.calRewardFunction(sample, settarget)
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

    def run(self, LOAD):
        maxIter = self.maxIter
        numDimOfSample = self.numDimOfSample
        numSamples = self.numSamples
        numTrials = self.numTrials
        #global settarget
        np.random.seed(1)
        allgoals = np.random.rand(numTrials,2)*15
        #self.csvwr.writecsv(currentdir+'/goal.csv', allgoals)

        R_mean = np.zeros(maxIter)
        R_std = np.zeros(maxIter)

        R_old = np.zeros(numSamples)
        R_mean_storage = np.zeros((maxIter, numTrials))
        Mu_w = np.zeros(numDimOfSample)
        Sigma_w = np.eye(numDimOfSample) * 1e6
        #settarget = np.random.rand(1, 2)[0] * 15
        X = np.empty(shape=(0, 2))
        Y = np.empty(shape=(0, 15))

        if LOAD:
            X = self.csvwr.readcsv(currentdir + '/inputX.csv')
            Y = self.csvwr.readcsv(currentdir + '/outputY.csv')
            X = np.array(X)
            Y = np.array(Y)
            print(X.shape, Y)
            m_load = GPy.models.GPRegression(X, Y, initialize=False)
            m_load.update_model(False)  # do not call the underlying expensive algebra on load
            m_load.initialize_parameter()  # Initialize the parameters (connect the parameters up)
            m_load[:] = np.load('model_save.npy')  # Load the parameters
            m_load.update_model(True)  # Call the algebra only once
            # print(m_load)
            display(m_load)
        for t in range(0, numTrials-1):
            settarget = allgoals[t]
            if LOAD:
                y, ysigma = m_load.predict(Xnew=np.array([settarget]))
                Mu_w = y[0]
                print('final : ', y, ysigma)
            if not LOAD:
                R_mean_storage = np.zeros((maxIter, numTrials))
                Mu_w = np.zeros(numDimOfSample)
                Sigma_w = np.eye(numDimOfSample) * 1e6
            print('target : ', settarget)
            print('trials No. : ', t)
            for k in range(0, maxIter):
                R, theta, traj = self.calculate_reward_and_theta(Mu_w, Sigma_w, settarget)
                # plot end config of sampled trajectories
                self.pjoint_global_update()
                plt.axis(([-30, 30, -30, 30]))
                plt.grid()
                plt.ion()
                plt.plot(self.px[0], self.py[0],'b.-')
                plt.plot(settarget[0], settarget[1], 'ro')
                plt.pause(0.00001)
                plt.cla()
                disx = self.px[0][-1] - settarget[0]
                disy = self.py[0][-1] - settarget[1]
                dis = sqrt(disx**2 + disy**2)
                # print('dis',dis)
                if dis < 1e-2:
                    print('iteration stop at ', k)
                    break

                #if np.linalg.norm(np.mean(R_old) - np.mean(R)) < 1e-3:
                #    break
                w = self.calculate_w(R, theta)
                Mu_w, Sigma_w = self.update_omega(w, theta)
                Sigma_w += np.eye(numDimOfSample)
                mR = np.mean(R)
                R_mean_storage[k, t] = mR
                R_old = R
                if k == maxIter and t == numTrials:
                    print(np.mean(R))
            print('1', traj)
            print('2', Sigma_w)
            print('3', Mu_w)
            print('start trajectory of trial ', t)
            # plot trajectory of last iteration
            for j in range(traj.shape[0]-1):
                self.jointPositions(traj[j + 1,::2])
                self.pjoint_global_update()
                #pos[i + 1, :] = self.fKinematics(q[i + 1, ::2])
                plt.axis(([-30, 30, -30, 30]))
                plt.grid()
                plt.ion()
                plt.plot(self.px[0], self.py[0], 'k.-')
                plt.plot(settarget[0], settarget[1], 'ro')
                plt.pause(0.000001)
                plt.cla()
            '''
            X = np.vstack((X, np.array(settarget)))
            print(X.shape, X, settarget)
            Y = np.vstack((Y, np.array(Mu_w)))
            print(Y.shape, Y)
            kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
            m = GPy.models.GPRegression(X, Y, kernel)
            m.optimize(messages=True)
            display(m)
            #settarget = np.random.rand(1, 2)[0] * 15
            #print("new target :", settarget, np.array([settarget]))
            
            if t<numTrials-1:
                y, ysigma = m.predict(Xnew=np.array([allgoals[t+1]]))
                Mu_w = y[0]
                print('predict mu : ', Mu_w, y, ysigma)
        
        self.csvwr.writecsv(currentdir+'/inputX.csv', X)
        self.csvwr.writecsv(currentdir+'/outputY.csv', Y)
        # let X, Y be data loaded above
        # Model creation:
        # m = GPy.models.GPRegression(X, Y)
        # m.optimize()
        # 1: Saving a model:
        np.save('model_save.npy', m.param_array)
        # 2: loading a model
        # Model creation, without initialization:
        m_load = GPy.models.GPRegression(X, Y, initialize=False)
        m_load.update_model(False)  # do not call the underlying expensive algebra on load
        m_load.initialize_parameter()  # Initialize the parameters (connect the parameters up)
        m_load[:] = np.load('model_save.npy')  # Load the parameters
        m_load.update_model(True)  # Call the algebra only once
        #print(m_load)
        display(m_load)
        y, ysigma = m.predict(Xnew=np.array([settarget]))
        print('final : ', y, ysigma)
        '''
        R_mean = np.mean(R_mean_storage, axis=1)
        R_std = np.sqrt(np.diag(np.cov(R_mean_storage)))
        print("Average return of final policy: ")
        print(R_mean[-1])
        print("\n")

if __name__ == '__main__':
    start_time = time.time()
    test = SimEnv3Joints()
    load = 0
    test.run(load)
    runningtime = time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))
