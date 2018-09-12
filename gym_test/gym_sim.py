#import tensorflow as tf
import gym
import numpy as np
import mujoco_py
from gym.envs.robotics import fetch_env
from gym.spaces import Dict


env = gym.make('FetchReach-v0')
print(type(env))
env.seed(1111)
a = env.observation_space

print('ob space ', a)
print((env.action_space.shape))
print(env.action_space.high, env.action_space.low)
#env.reward_type = 'dense'

#env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal'])
observation = env.reset()
print('joint_positions',observation['observation'])

model = mujoco_py.load_model_from_path("/home/zheng/zzc_ws/gym/gym/envs/robotics/assets/fetch/reach.xml")
sim = mujoco_py.MjSim(model)
print(sim.data.get_site_xpos('robot0:grip'))
# ['robot0:slide0', 'robot0:slide1', 'robot0:slide2', 'robot0:torso_lift_joint', 'robot0:head_pan_joint', 'robot0:head_tilt_joint',
# 'robot0:shoulder_pan_joint', 'robot0:shoulder_lift_joint', 'robot0:upperarm_roll_joint', 'robot0:elbow_flex_joint', 'robot0:forearm_roll_joint',
# 'robot0:wrist_flex_joint', 'robot0:wrist_roll_joint', 'robot0:r_gripper_finger_joint', 'robot0:l_gripper_finger_joint']

if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        print(names)
        print(np.array([sim.data.get_joint_qpos(name) for name in names]),
              np.array([sim.data.get_joint_qvel(name) for name in names]),
              sim.data.get_site_xpos('robot0:grip'),
              sim.data.get_site_xvelp('robot0:grip'))
observation = env.reset()
s = observation
print('sdim', s['observation'].shape[0] + s['achieved_goal'].shape[0] + s['desired_goal'].shape[0], np.concatenate([observation['achieved_goal'], observation['observation']]))
#print('lll',observation['joint_positions'])
observation = env.reset()
#print('lll',observation['joint_positions'])
for i_episode in range(1):
    env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(i_episode, 'xxx episode observation : ', observation['observation'])
    #action = [0,0,0,0]
    for t in range(10):
        env.render()
        #observation['desired_goal'] = np.array([0, 0, 0.5])
        #print('1st observation : ', observation)
        #print('action', action)
        # action = env.action_space.sample()
        action = np.zeros(4)
        #action = [0.09762701, 0.43037874, 0.20552675, 0.08976637]
        #action = action
        #print('action2', action)
        observation, reward, done, info = env.step(action)
       # print('ob2', observation['joint_positions'])
        #print('reward1', reward, np.concatenate([observation['achieved_goal'], observation['desired_goal']]))
        #reward == env.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)
        '''
        if sim.data.qpos is not None and sim.model.joint_names:
            names = [n for n in sim.model.joint_names if n.startswith('robot')]
            print(names)
            print(np.array([sim.data.get_joint_qpos(name) for name in names]), np.array([sim.data.get_joint_qvel(name) for name in names]))
        '''
        #print('reward2', reward)
        #print('observation : ', observation['desired_goal'], '\n', 'reward : ', reward, '\n', 'done : ', done, '\n', 'info : ', info)
        print(' first observation : ', observation['achieved_goal'])
        #action[:3] = observation['desired_goal'] - observation['achieved_goal']
        #action = [action[0],action[1],action[2], 0]
        #print('ob para : ', action, np.dot(action, action))
        action = [-0.09762701, -0.43037874, -0.20552675, -0.08976637]
        action = np.zeros(4)
        #print('action3', action)
        observation, reward, done, info = env.step(action)
        #print('observation : ', observation['desired_goal'], '\n', 'reward : ', reward, '\n', 'done : ', done, '\n',  'info : ', info)
        print( ' second observation : ', observation['achieved_goal'])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            #break