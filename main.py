from agent import train_dqn
from snake_env import Snake
import rdflib
from experiments import run_experiments
from helpers import split_res
import numpy as np
import random
from sklearn.linear_model import LinearRegression


def get_default_parameters():
    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    ep = 20

    return ep, params


def try_model_with_parameters(ep, params):
    env = Snake()
    sum_of_rewards = train_dqn(ep, env, params)
    x = np.arange(1, ep + 1, 1)
    clf = LinearRegression()
    yreshaped = x.reshape(-1, 1)
    clf.fit(yreshaped, np.array(sum_of_rewards).reshape(-1))
    pred = clf.predict(yreshaped)

    return pred[ep - 1]


def random_redefine_parameters():
    print('[INFO] Random redefining parameters...')
    params = dict()
    params['name'] = None
    params['epsilon'] = random.uniform(0.1, 1)
    params['gamma'] = random.uniform(0.1, 1)
    params['batch_size'] = random.randint(10, 1000)
    params['epsilon_min'] = random.uniform(0.001, 0.2)
    params['epsilon_decay'] = random.uniform(0.01, 0.96)
    params['learning_rate'] = random.uniform(0.0001, 0.0003)
    params['layer_sizes'] = [128, 128, 128]

    ep = random.randint(10, 100)

    return ep, params


def get_parameters_from_kb(ep, params, kb):
    for row in kb.query("SELECT ?s WHERE { mo:episodes_info  mp:best_is ?s .}"):
        ep = int(split_res(row.s))
        print('episodes', ep)

    for row in kb.query("SELECT ?s WHERE { mo:batchsize_info  mp:best_is ?s .}"):
        batchsize = int(split_res(row.s))
        print('batchsize', batchsize)
        params['batch_size'] = batchsize

    for row in kb.query("SELECT ?s WHERE { mo:epsilon_info  mp:best_is ?s .}"):
        epsilon = int(split_res(row.s))
        print('epsilon', epsilon)
        params['epsilon'] = epsilon

    for row in kb.query("SELECT ?s WHERE { mo:learning_rate_info  mp:best_is ?s .}"):
        learning_rate = int(split_res(row.s))
        print('learning_rate', learning_rate)
        params['learning_rate'] = learning_rate

    for row in kb.query("SELECT ?s WHERE { mo:gamma_info  mp:best_is ?s .}"):
        gamma = int(split_res(row.s))
        print('gamma', gamma)
        params['gamma'] = gamma

    for row in kb.query("SELECT ?s WHERE { mo:epsilon_min  mp:best_is ?s .}"):
        epsilon_min = int(split_res(row.s))
        print('epsilon_min', epsilon_min)
        params['epsilon_min'] = epsilon_min

    for row in kb.query("SELECT ?s WHERE { mo:epsilon_decay  mp:best_is ?s .}"):
        epsilon_decay = int(split_res(row.s))
        print('epsilon_decay', epsilon_decay)
        params['epsilon_decay'] = epsilon_decay

    return ep, params


def check_is_need_redefine_parameters(attempt_score, kb):
    for row in kb.query("SELECT ?s WHERE { mo:bad_result mp:less_than ?s .}"):
        target_score_knowledge = split_res(row.s)
        print('target_score_knowledge', target_score_knowledge)
        if target_score_knowledge:
            for row in kb.query("SELECT ?s WHERE { mo:%(name)s mp:good_is ?s .}" % {"name": target_score_knowledge}):
                score_metric = int(split_res(row.s))
                print('score_metric', score_metric)
                return attempt_score < score_metric
    return False


if __name__ == "__main__":
    g = rdflib.Graph()
    g.load("./knowledge_base.n3", format="n3")

    ep, params = get_default_parameters()
    run_experiments(ep, params, g)

    ep, params = get_parameters_from_kb(ep, params, g)

    score_metric = 0
    for row in g.query("SELECT ?s WHERE { mo:score mp:good_is ?s .}"):
        score_metric = int(split_res(row.s))
        print('score_metric:', score_metric)

    max_attempts = 0
    for row in g.query("SELECT ?s WHERE { mo:attempts mp:max_is ?s .}"):
        max_attempts = int(split_res(row.s))
        print('max_attempts:', max_attempts)

    curr_num_attempt = 0
    best_attempt_score = try_model_with_parameters(ep, params)
    is_need_redefine_parameters = check_is_need_redefine_parameters(best_attempt_score, g)

    while is_need_redefine_parameters:
        print('[INFO] Need to redefine parameters. Experiments is running...')
        if curr_num_attempt >= max_attempts:
            for row in g.query("SELECT ?s WHERE { mo:attempts mp:needs ?s .}"):
                action = split_res(row.s)
                print('action', action)
                if action == 'run_random_redefine_parameters':
                    for row in g.query("SELECT ?s WHERE { mo:run_random_redefine_parameters mp:calls ?s .}"):
                        method_to_call = split_res(row.s)
                        print('method_to_call', method_to_call)
                        ep, params = globals()[method_to_call]()
                        best_attempt_score = try_model_with_parameters(ep, params)
                        is_need_redefine_parameters = check_is_need_redefine_parameters(best_attempt_score, g)
            if not is_need_redefine_parameters:
                break
            curr_num_attempt = 0

        for row in g.query("SELECT ?s WHERE { mo:bad_result mp:need_to ?s .}"):
            action = split_res(row.s)
            print('action', action)
            if action == 'redefine_parameters':
                run_experiments(ep, params, g)
        ep, params = get_parameters_from_kb(ep, params, g)
        best_attempt_score = try_model_with_parameters(ep, params)
        is_need_redefine_parameters = check_is_need_redefine_parameters(best_attempt_score, g)
        curr_num_attempt += 1

    print('[INFO] Model satisfies the requirements.')

    # SAVE KNOWLEDGE BASE BACKUP WITH RESULTS TO A FILE
    g.serialize(destination='output_kb.n3', format="turtle")
