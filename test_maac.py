import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
import time


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(1804)
    np.random.seed(1804)
    # initialize E parallel environments with N agents
    env = make_parallel_env(config.env_id, config.n_rollout_threads, 1804)
    model = AttentionSAC.init_from_save('model.pt')
    # model = AttentionSAC.init_from_env(env,
    #                                    tau=config.tau,
    #                                    pi_lr=config.pi_lr,
    #                                    q_lr=config.q_lr,
    #                                    gamma=config.gamma,
    #                                    pol_hidden_dim=config.pol_hidden_dim,
    #                                    critic_hidden_dim=config.critic_hidden_dim,
    #                                    attend_heads=config.attend_heads,
    #                                    reward_scale=config.reward_scale)
    # initialize replay buffer D
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    
    # T_update 
    t = 0
    max_step = 0
    max_time = 0
    total_step = np.zeros(model.nagents)
    total_time = np.zeros(model.nagents)
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')
        
        success = np.zeros(
            (config.n_rollout_threads, model.nagents), dtype=bool)
        steps = np.zeros(
            (config.n_rollout_threads, model.nagents))
        time_cost = np.zeros(
            (config.n_rollout_threads, model.nagents))
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            
            start = time.clock()
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=False)
            end = time.clock()
            per_time_cost = end-start
            
            
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
        
            # calculate steps
            success = np.logical_or(success, dones)
            # steps += dones
            steps += np.logical_not(dones)
            time_cost += np.logical_not(dones) * per_time_cost
            
                
            # store transitions for all env in replay buffer
            # replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            
            # T_update = T_update + E
            t += config.n_rollout_threads
            
            # if (len(replay_buffer) >= max(config.pi_batch_size, config.q_batch_size) and
            #     (t % config.steps_per_update) < config.n_rollout_threads):
            #     if config.use_gpu:
            #         model.prep_training(device='gpu')
            #     else:
            #         model.prep_training(device='cpu')
            #     for u_i in range(config.num_critic_updates):
            #         sample = replay_buffer.sample(config.q_batch_size,
            #                                       to_gpu=config.use_gpu)
            #         model.update_critic(sample, logger=logger)
            #     for u_i in range(config.num_pol_updates):
            #         sample = replay_buffer.sample(config.pi_batch_size,
            #                                       to_gpu=config.use_gpu)
            #         model.update_policies(sample, logger=logger)
            #     model.update_all_targets()
            #     # for u_i in range(config.num_updates):
            #     #     sample = replay_buffer.sample(config.batch_size,
            #     #                                   to_gpu=config.use_gpu)
            #     #     model.update_critic(sample, logger=logger)
            #     #     model.update_policies(sample, logger=logger)
            #     #     model.update_all_targets()
            model.prep_rollouts(device='cpu')
                
        # ep_dones = np.mean(success, axis=0)
        # ep_steps = 1 - np.mean(steps / config.episode_length, axis=0)
        # ep_mean_step
        
        # ep_rews = replay_buffer.get_average_rewards(
        #     config.episode_length * config.n_rollout_threads)
        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        # for a_i, a_ep_done in enumerate(ep_dones):
            # logger.add_scalar('agent%i/mean_episode_dones' % a_i, a_ep_done, ep_i)
        # for a_i, a_ep_step in enumerate(ep_steps):
            # logger.add_scalar('agent%i/mean_episode_steps' % a_i, a_ep_step, ep_i)

        total_step += np.mean(steps, axis=0)
        total_time += np.mean(time_cost, axis=0)
        
        max_step += np.max(steps)
        max_time += np.max(time_cost)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            # os.makedirs(run_dir / 'incremental', exist_ok=True)
            # model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            # model.save(run_dir / 'model.pt')
    
    mean_step = total_step / (100/config.n_rollout_threads)
    mean_time = total_time / (100/config.n_rollout_threads)
    max_time /= 100/config.n_rollout_threads
    max_step /= 100/config.n_rollout_threads
    
    print('; '.join([f'{chr(65 + i)} Mean Step:{mean_step[i]}, Mean Time:{mean_time[i]}'
                    for i in range(model.nagents)]))
    print('Mean Max Step:{}, Mean Max Time Cost:{}'.format(max_step, max_time))
    # model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",
                        default='simple_route_plan',
                        help="Name of environment")
    parser.add_argument("--model_name",
                        default='test_task',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    # parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    # parser.add_argument("--num_updates", default=4, type=int,
    #                     help="Number of updates per update cycle")
    parser.add_argument("--num_critic_updates", default=4, type=int,
                        help="Number of critic updates per update cycle")
    parser.add_argument("--num_pol_updates", default=4, type=int,
                        help="Number of policy updates per update cycle")
    # parser.add_argument("--batch_size",
    #                     default=1024, type=int,
    #                     help="Batch size for training")
    parser.add_argument("--pi_batch_size",
                        default=1024, type=int,
                        help="Batch size for policy training")
    parser.add_argument("--q_batch_size",
                        default=1024, type=int,
                        help="Batch size for critic training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)
