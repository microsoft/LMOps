import urllib3
import math
import numpy as np
import random
from tqdm import tqdm
import concurrent.futures
import requests


class SuccessiveHalvingEvaluator:
    """ Successive Halving Evaluator """
    def __init__(self, config):
        self.config = config

    def __call__(self, prompts, exs, task, predictor, scorer, rounds=40,
    num_prompts_per_round=10, samples_per_eval=5, max_threads=1, verbose=False, budget=None):

        out_ranks = [-1] * len(prompts)
        prompt2idx = {p: i for i, p in enumerate(prompts)}
        idx2prompts = {i: p for i, p in enumerate(prompts)}

        num_rounds = len(prompts) - self.config['beam_size']
        if budget is None:
            budget = self.config['eval_budget']
        n = len(prompts)
        S = prompts

        for r in range(0, math.ceil(math.log2(n))):
            t_r = math.floor(budget / (len(S) * math.ceil(math.log2(n))))

            sample = random.sample(exs, min(len(exs), (t_r)))

            while True:
                try:
                    scores = scorer(predictor, S, sample, max_threads=max_threads)
                    break
                except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError, urllib3.exceptions.MaxRetryError):
                    pass

            average = np.mean(scores)
            for score, prompt in zip(scores, S):
                if score < average:
                    out_ranks[prompt2idx[prompt]] = r

            S = [prompt for (score, prompt) in zip(scores, S) if score >= average]

        n_top_rank = sum(1 for x in out_ranks if x == -1)
        if n_top_rank < self.config['beam_size']:
            # TODO get the boundary of the beam
            target_prompts = [
                idx2prompts[i] for i, rank 
                in enumerate(out_ranks) if rank == r
            ]

        r = r + 1
        for i in range(len(out_ranks)):
            if out_ranks[i] == -1:
                out_ranks[i] = r

        return out_ranks


class SuccessiveRejectsEvaluator:
    """ Successive Rejects Evaluator """
    def __init__(self, config):
        self.config = config

    def __call__(self, prompts, exs, task, predictor, scorer, rounds=40,
    num_prompts_per_round=10, samples_per_eval=5, max_threads=1, verbose=False):
        assert self.config['evaluator'] in {'sr', 's-sr'}, f'unk evaluator: {self.config["evaluator"]}'

        out_ranks = [-1] * len(prompts)
        idx2prompt = {i: p for i, p in enumerate(prompts)}

        # only run the algo until the beam is full
        num_rounds = len(prompts) - self.config['beam_size']

        if self.config['evaluator'] == 's-sr':
            # calculate the number of datapoints to use per rejection test
            samples_per_round = math.ceil(self.config['eval_budget'] / (num_rounds * num_prompts_per_round))
            if samples_per_round == 0:
                raise Exception(f"not enough budget for s-sr!budget: {self.config['eval_budget']}")

        elif self.config['evaluator'] == 'sr':
            K = len(prompts) - self.config['beam_size'] # if its on the beam we dont care about order
            log_bar_K = 0.5 + sum([1.0/i for i in range(2, K+1)])
            n_prev_k = 0

        current_usage = 0
        ri = 1
        with tqdm(total=len(idx2prompt), desc='sr') as pbar:
            while True:
                if len(idx2prompt) <= self.config['beam_size']:
                    break

                if self.config['evaluator'] == 's-sr':
                    selected_data = random.sample(exs, samples_per_round)
                    selected_idxs, selected_prompts = list(zip(*random.sample(
                        idx2prompt.items(), min(num_prompts_per_round, len(idx2prompt)))))

                elif self.config['evaluator'] == 'sr':
                    selected_idxs, selected_prompts = list(zip(*idx2prompt.items()))
                    n_k = (1.0 / log_bar_K) * ((self.config['eval_budget'] - K) / (K + 1 - ri))
                    samples_per_round = int(n_k - n_prev_k)
                    samples_per_round = max(4, samples_per_round)
                    selected_data = random.sample(exs, min(len(exs), samples_per_round))
                    n_prev_k = n_k
                    if len(selected_data) == 0:
                        raise Exception(f'not enough budget for SR! budget: {self.config["eval_budget"]}')

                while True:
                    try:
                        scores = scorer(predictor, selected_prompts, selected_data, max_threads=max_threads)
                        break
                    except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError, urllib3.exceptions.MaxRetryError):
                        pass

                current_usage += (len(selected_prompts) * len(selected_data))
                ri += 1
                min_idx = scores.index(min(scores))

                idxs_to_remove = [selected_idxs[min_idx]]

                for i in idxs_to_remove:
                    del idx2prompt[i] # reject the selected arm 
                    out_ranks[i] = ri # higher score is better so increase as survives

                pbar.update(1)

        # fill in the beam with default values
        ri += 1
        for i in range(len(out_ranks)):
            if out_ranks[i] == -1:
                out_ranks[i] = ri

        return out_ranks



class UCBBandits:
    """ Upper Confidence Bound Bandits """
    def __init__(self, num_prompts, num_samples=5, c=1.0, mode='ucb'):
        self.c = c
        assert mode in {'ucb', 'ucb-e'}
        self.mode = mode
        self.num_prompts = num_prompts
        self.num_samples = num_samples
        self.reset()

    def update(self, chosen, scores):
        for i, score in zip(chosen, scores):
            self.counts[i] += self.num_samples
            self.scores[i] += score * self.num_samples

    def reset(self):
        self.counts = np.zeros(self.num_prompts)
        self.scores = np.zeros(self.num_prompts)

    def get_scores(self):
        # Some counts may be 0, so we need to avoid division by 0.
        return np.divide(self.scores, self.counts, out=np.zeros_like(self.scores), where=self.counts != 0)

    def choose(self, n, t):
        if np.sum(self.counts) == 0:
            # If all counts are 0, choose randomly.
            return random.sample(range(self.num_prompts), n)
        scores = self.get_scores()
        counts = self.counts + 1e-3
        if self.mode == 'ucb':
            ucb_scores = scores + self.c * np.sqrt(np.log(t) / counts)
        elif self.mode == 'ucb-e':
            ucb_scores = scores + self.c * np.sqrt(self.c / counts)

        # Choose the prompts with the highest UCB scores
        return np.argsort(ucb_scores)[::-1][:n]

    def get_infos(self):
        return self.counts


class UCBBanditEvaluator:
    """ Upper Confidence Bound Evaluator"""
    def __init__(self, config):
        self.config = config

    def __call__(self, prompts, exs, task, predictor, scorer, 
               rounds=40, num_prompts_per_round=10, samples_per_eval=5, max_threads=1, verbose=True):
        assert self.config['evaluator'] in {'ucb', 'ucb-e'}, f'unk evaluator: {self.config["evaluator"]}'
        bandit_algo = UCBBandits(
            len(prompts), num_samples=samples_per_eval,
            mode=self.config['evaluator'],
            c=self.config['c']
        )
        
        def data_sampler(l):
            return random.sample(l, samples_per_eval)

        num_prompts_per_round = min(num_prompts_per_round, len(prompts))

        for ri in tqdm(range(rounds), desc=f'Evaluating {len(prompts)} prompts'):
            # Sample the prompts
            sampled_prompts_idx = bandit_algo.choose(num_prompts_per_round, ri)
            sampled_prompts = [prompts[i] for i in sampled_prompts_idx]
            sampled_data = data_sampler(exs)
            while True:
                try:
                    scores = scorer(predictor, sampled_prompts, sampled_data, max_threads=max_threads)
                    break
                except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError, urllib3.exceptions.MaxRetryError):
                    pass
            bandit_algo.update(sampled_prompts_idx, scores)
            
        return bandit_algo.get_scores().tolist()


class BruteForceEvaluator:
    """ Brute Force Evaluator """
    def __init__(self, config):
        self.config = config

    def __call__(self, prompts, exs, task, predictor, scorer,
rounds=40, num_prompts_per_round=10, c=2.0, samples_per_eval=5, max_threads=1, verbose=True):
        sample_size = min(len(exs), int(self.config['eval_budget'] / len(prompts)))
        eval_exs = random.sample(exs, sample_size)

        while True:
            try:
                scores = scorer(predictor, prompts, eval_exs, max_threads=max_threads)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError, urllib3.exceptions.MaxRetryError):
                pass
        return scores
