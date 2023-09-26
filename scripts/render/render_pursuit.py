#!/usr/bin/env python
import sys
sys.path.append("../")
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.pursuit.pursuit_env import PursuitEnv
from envs.env_wrappers import PursuitSubprocVecEnv, PursuitDummyVecEnv

"""Train script for MPEs."""

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "pursuit":
                env = PursuitEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return PursuitDummyVecEnv([get_env_fn(0)])
    else:
        return PursuitSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int,
                        default=8, help="number of players")
    parser.add_argument('--obs_range', type=int,
                        default=7, help="observation range")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    if all_args.seed_specify:
        all_args.seed=all_args.seed
    else:
        all_args.seed=np.random.randint(1000,10000)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    cur_dir=Path(os.path.join(os.path.dirname(__file__)))
    result_dir=os.path.join(cur_dir.parent.absolute().parent.absolute(), "results")
    
    run_dir = Path(result_dir) / all_args.env_name / all_args.algorithm_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))


    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.pursuit_runner import PursuitRunner as Runner
    else:
        raise Exception('Non parameter sharing vesion is not implemented yet!') 

    runner = Runner(config)
    runner.render()
    
    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
