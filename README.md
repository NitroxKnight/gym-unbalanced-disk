# gym-unbalanced-disk

Installation

```
git clone git@github.com:upb-lea/gym-electric-motor.git 
cd gym-electric-motor
pip install -e .
```


basic use

```
import gym, gym_unbalanced_disk, time


env = gym.make('unbalanced-disk-v0')

obs = env.reset()
for i in range(200):
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs,reward)
    env.render()
    time.sleep(1/24)
    if done:
      obs = env.reset()
env.close()
```
