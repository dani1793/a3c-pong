import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import prepro
from model import ActorCritic
from pong import Pong
from simple_ai import PongAi



def test(rank, args, shared_model, counter, optimizer):
    torch.manual_seed(args.seed + rank)

    env = Pong()
    # env.seed(args.seed + rank)

    model = ActorCritic(1, 3)
    
    opponent = PongAi(env, 2)
    
    model.eval()

    state = prepro(env.reset()[0])
    
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        env.render()
        
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            inputTensor = state.unsqueeze(0).unsqueeze(0);
            value, logit, (hx, cx) = model((inputTensor, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        
        action2 = opponent.get_action()
        
        (state, obs2), (reward, reward2), done, info = env.step((action[0,0], action2))
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward
        state = prepro(state)
        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == 5000:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            save_checkpoint(shared_model,optimizer, "checkpoint/num steps {}-episode_reward {}-episode_length {}.pth".format(
                counter.value,
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = prepro(env.reset()[0])
            time.sleep(60)

        state = torch.from_numpy(state)
        
def save_checkpoint(model, optimizer, filename='/output/checkpoint.pth.tar'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, filename)