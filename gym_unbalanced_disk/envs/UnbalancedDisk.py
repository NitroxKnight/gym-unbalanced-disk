
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path

class UnbalancedDisk(gym.Env):
    def __init__(self):
        self.g = 9.80155078791343
        self.J = 0.000244210523960356
        self.Km = 10.5081817407479
        self.I = 0.0410772235841364
        self.M = 0.0761844495320390
        self.tau = 0.397973147009910
        self.dt = 0.025

        # self.action_space = spaces.Box(low=[-3],high=[3],shape=(1,))
        self.action_space = spaces.Discrete(5)
        low = [-1,-1,-float('inf')]
        high = [1,1,float('inf')]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(3,))

        self.reward_fun = lambda self: np.exp(-((self.th%(2*np.pi))-np.pi)**2/(2*(np.pi/10)**2))
        self.viewer = None
        self.u = 0

    def step(self,action):
        #convert to u
        # self.u = [-3.5,-1,0,1,3.5][action]
        self.u = [-4,-1,0,1,4][action] #[-3,-1,0,1,3][action]
        f = lambda t,y: [y[1],-self.M*self.g*self.I/self.J*np.sin(y[0]) + self.u*self.Km/self.tau - 1/self.tau*y[1]]
        sol = solve_ivp(f,[0,self.dt],[self.th,self.omega])
        self.th, self.omega = sol.y[:,-1]
        reward = self.reward_fun(self)

        return self.get_obs(), reward, False, {}
        

    def reset(self,seed=None):
        self.th = np.random.normal(loc=0,scale=0.01)
        self.omega = np.random.normal(loc=0,scale=0.001)
        return self.get_obs()

    def get_obs(self):
        th_noise = self.th + np.random.normal(loc=0,scale=0.001)
        omega_noise = self.omega + np.random.normal(loc=0,scale=0.001)
        return np.array([np.sin(th_noise), np.sin(omega_noise), omega_noise])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.th - np.pi / 2)
        if self.u:
            self.imgtrans.scale = (-self.u / 2, np.abs(self.u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
