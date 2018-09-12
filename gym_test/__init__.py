from gym.envs.registration import register

register(
    id='arm3joints-v0',
    entry_point='gym_pre_arms.envs:Arm3jointsEnv',
)
