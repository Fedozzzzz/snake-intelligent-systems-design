from agent import train_dqn
from snake_env import Snake
import numpy as np
from experimental.plot_script import plot_result
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from rdflib import URIRef, Literal
from helpers import split_res


def run_experiment(experiment_name, params):
    if experiment_name == 'run_episodes_num_experiment':
        global_results = dict()
        for ep in range(5, 50, 5):
            results = dict()
            env = Snake()
            sum_of_rewards = train_dqn(ep, env, params)
            global_results[ep] = max(sum_of_rewards)
            results[params['name']] = sum_of_rewards

            plot_result(results, direct=True, k=20)
        return max(global_results, key=global_results.get)

    elif experiment_name == 'run_batchsize_experiment':
        global_results = dict()
        for batchsz in [1, 10, 100, 1000]:
            print(batchsz)
            params['batch_size'] = batchsz
            params['name'] = f'Batchsize {batchsz}'
            env = Snake()
            ep = 20
            sum_of_rewards = train_dqn(ep, env, params)

            x = np.arange(1, ep + 1, 1)
            clf = LinearRegression()
            yreshaped = x.reshape(-1, 1)
            clf.fit(yreshaped, np.array(sum_of_rewards).reshape(-1))
            pred = clf.predict(yreshaped)
            plt.plot(x, pred)
            global_results[batchsz] = pred[ep - 1]
        return max(global_results, key=global_results.get)

    elif experiment_name == 'run_epsilon_experiment':
        global_results = dict()
        plt.figure(figsize=(20, 10))
        labels = []
        for epsilon in np.arange(0.1, 1.1, 0.1):
            print('epsilon:', epsilon)
            params['epsilon'] = epsilon
            params['name'] = f'epsilon {epsilon}'
            labels.append(params['name'])
            env = Snake()
            ep = 20
            sum_of_rewards = train_dqn(ep, env, params)

            x = np.arange(1, ep + 1, 1)
            clf = LinearRegression()
            yreshaped = x.reshape(-1, 1)
            clf.fit(yreshaped, np.array(sum_of_rewards).reshape(-1))
            pred = clf.predict(yreshaped)
            plt.plot(x, pred)
            global_results[epsilon] = pred[ep - 1]
        plt.legend(labels)
        plt.show()
        return max(global_results, key=global_results.get)

    elif experiment_name == 'run_learning_rate_experiment':
        print('[INFO] "learning_rate" experiment is running...')
        global_results = dict()
        plt.figure(figsize=(20, 10))
        labels = []
        for learning_rate in np.arange(0.00005, 0.0005, 0.00005):
            print('learning_rate:', learning_rate)
            params['learning_rate'] = learning_rate
            params['name'] = f'epsilon {learning_rate}'
            labels.append(params['name'])
            env = Snake()
            ep = 15
            sum_of_rewards = train_dqn(ep, env, params)

            x = np.arange(1, ep + 1, 1)
            clf = LinearRegression()
            yreshaped = x.reshape(-1, 1)
            clf.fit(yreshaped, np.array(sum_of_rewards).reshape(-1))
            pred = clf.predict(yreshaped)
            plt.plot(x, pred)
            global_results[learning_rate] = pred[ep - 1]
        plt.legend(labels)
        plt.show()
        return max(global_results, key=global_results.get)

    elif experiment_name == 'run_gamma_experiment':
        print('[INFO] "Gamma" experiment is running...')
        global_results = dict()
        plt.figure(figsize=(20, 10))
        labels = []
        for gamma in np.arange(0.05, 1.05, 0.05):
            print('gamma:', gamma)
            params['gamma'] = gamma
            params['name'] = f'gamma {gamma}'
            labels.append(params['name'])
            env = Snake()
            ep = 15
            sum_of_rewards = train_dqn(ep, env, params)

            x = np.arange(1, ep + 1, 1)
            clf = LinearRegression()
            yreshaped = x.reshape(-1, 1)
            clf.fit(yreshaped, np.array(sum_of_rewards).reshape(-1))
            pred = clf.predict(yreshaped)
            plt.plot(x, pred)
            global_results[gamma] = pred[ep - 1]
        plt.legend(labels)
        plt.show()
        return max(global_results, key=global_results.get)

    elif experiment_name == 'run_epsilon_min_experiment':
        print('[INFO] "epsilon_min" experiment is running...')
        global_results = dict()
        plt.figure(figsize=(20, 10))
        labels = []
        for epsilon_min in np.arange(0.8, 1.0, 0.005):
            print('epsilon_min:', epsilon_min)
            params['epsilon_min'] = epsilon_min
            params['name'] = f'gamma {epsilon_min}'
            labels.append(params['name'])
            env = Snake()
            ep = 15
            sum_of_rewards = train_dqn(ep, env, params)

            x = np.arange(1, ep + 1, 1)
            clf = LinearRegression()
            yreshaped = x.reshape(-1, 1)
            clf.fit(yreshaped, np.array(sum_of_rewards).reshape(-1))
            pred = clf.predict(yreshaped)
            plt.plot(x, pred)
            global_results[epsilon_min] = pred[ep - 1]
        plt.legend(labels)
        plt.show()
        return max(global_results, key=global_results.get)

    elif experiment_name == 'run_epsilon_decay_experiment':
        print('[INFO] "epsilon_decay" experiment is running...')
        global_results = dict()
        for epsilon_decay in np.arange(0.5, 1.0, 0.01):
            print('epsilon_decay:', epsilon_decay)
            params['epsilon_decay'] = epsilon_decay
            params['name'] = f'gamma {epsilon_decay}'
            env = Snake()
            ep = 5
            sum_of_rewards = train_dqn(ep, env, params)

            x = np.arange(1, ep + 1, 1)
            clf = LinearRegression()
            yreshaped = x.reshape(-1, 1)
            clf.fit(yreshaped, np.array(sum_of_rewards).reshape(-1))
            pred = clf.predict(yreshaped)
            global_results[epsilon_decay] = pred[ep - 1]
        return max(global_results, key=global_results.get)

    elif experiment_name == 'run_env_experiment':
        print('[INFO] env experiment is running...')
        env_infos = {
            'States: only walls': {'state_space': 'no body knowledge'},
            'States: direction 0 or 1': {'state_space': ''},
            'States: coordinates': {'state_space': 'coordinates'},
            'States: no direction': {'state_space': 'no direction'}}
        plt.figure(figsize=(20, 10))
        labels = []
        global_results = dict()
        for key in env_infos.keys():
            params['name'] = key
            labels.append(key)

            env_info = env_infos[key]
            print(env_info)
            env = Snake(env_info=env_info)
            ep = 20
            sum_of_rewards = train_dqn(ep, env, params)

            x = np.arange(1, ep + 1, 1)
            clf = LinearRegression()
            yreshaped = x.reshape(-1, 1)
            clf.fit(yreshaped, np.array(sum_of_rewards).reshape(-1))
            pred = clf.predict(yreshaped)
            plt.plot(x, pred)
            global_results[key] = pred[ep - 1]
        plt.legend(labels)
        plt.show()
        return max(global_results, key=global_results.get)


def run_experiments(ep, params, kb):
    print('[INFO] "learning_rate" experiment is running...')
    for row in kb.query("SELECT ?s WHERE { mo:learning_rate_experiment  mp:calls ?s .}"):
        field = split_res(row.s)
        print('field', field)
        best_learning_rate_value = run_experiment(field, params)
        print('[INFO] "learning_rate" experiment has finished. Best is', best_learning_rate_value)

        epsilon_info = URIRef('urn:myObjects:learning_rate_info')
        best_is = URIRef('urn:myPredicates:best_is')
        value = Literal(best_learning_rate_value)

        kb.add((epsilon_info, best_is, value))

    print('[INFO] "Epsilon" experiment is running...')
    for row in kb.query("SELECT ?s WHERE { mo:epsilon_experiment  mp:calls ?s .}"):
        field = split_res(row.s)
        print('field', field)
        best_epsilon_value = run_experiment(field, params)
        print('[INFO] "Epsilon" experiment has finished. Best is', best_epsilon_value)

        epsilon_info = URIRef('urn:myObjects:epsilon_info')
        best_is = URIRef('urn:myPredicates:best_is')
        value = Literal(best_epsilon_value)

        kb.add((epsilon_info, best_is, value))

    print('[INFO] Number of episodes experiment is running...')
    for row in kb.query("SELECT ?s WHERE { mo:episodes_num_experiment  mp:calls ?s .}"):
        field = split_res(row.s)
        print('field', field)
        best_episodes_value = run_experiment(field, params)
        print('[INFO] Number of episodes experiment has finished.')

        episodes_info = URIRef('urn:myObjects:episodes_info')
        best_is = URIRef('urn:myPredicates:best_is')
        value = Literal(best_episodes_value)

        kb.add((episodes_info, best_is, value))

    print('[INFO] Batch size experiment is running...')
    for row in kb.query("SELECT ?s WHERE { mo:batchsize_experiment  mp:calls ?s .}"):
        field = split_res(row.s)
        print('field', field)
        best_batchsize_value = run_experiment(field, params)
        print('[INFO] Batch size experiment has finished.')

        batchsize_info = URIRef('urn:myObjects:batchsize_info')
        best_is = URIRef('urn:myPredicates:best_is')
        value = Literal(best_batchsize_value)

        kb.add((batchsize_info, best_is, value))

    print('[INFO] Gamma experiment is running...')
    for row in kb.query("SELECT ?s WHERE { mo:gamma_experiment  mp:calls ?s .}"):
        field = split_res(row.s)
        print('field', field)
        best_gamma_value = run_experiment(field, params)
        print('[INFO] Gamma experiment has finished. best is ', best_gamma_value)

        gamma_info = URIRef('urn:myObjects:gamma_info')
        best_is = URIRef('urn:myPredicates:best_is')
        value = Literal(best_gamma_value)

        kb.add((gamma_info, best_is, value))

    print('[INFO] epsilon_min experiment is running...')
    for row in kb.query("SELECT ?s WHERE { mo:epsilon_min_experiment  mp:calls ?s .}"):
        field = split_res(row.s)
        print('field', field)
        best_epsilon_min_value = run_experiment(field, params)
        print('[INFO] epsilon_min experiment has finished. best is ', best_epsilon_min_value)

        epsilon_min_info = URIRef('urn:myObjects:epsilon_min')
        best_is = URIRef('urn:myPredicates:best_is')
        value = Literal(best_epsilon_min_value)

        kb.add((epsilon_min_info, best_is, value))

    print('[INFO] epsilon_decay experiment is running...')
    for row in kb.query("SELECT ?s WHERE { mo:epsilon_decay_experiment  mp:calls ?s .}"):
        field = split_res(row.s)
        print('field', field)
        best_epsilon_decay_value = run_experiment(field, params)
        print('[INFO] epsilon_decay experiment has finished. best is ', best_epsilon_decay_value)

        epsilon_decay_info = URIRef('urn:myObjects:epsilon_decay')
        best_is = URIRef('urn:myPredicates:best_is')
        value = Literal(best_epsilon_decay_value)

        kb.add((epsilon_decay_info, best_is, value))

    print('[INFO]  "Environment" experiment is running...')
    for row in kb.query("SELECT ?s WHERE { mo:env_experiment  mp:calls ?s .}"):
        field = split_res(row.s)
        print('field', field)
        best_env_value = run_experiment(field, params)
        print('[INFO] "Environment" experiment has finished.')

        batchsize_info = URIRef('urn:myObjects:env_info')
        best_is = URIRef('urn:myPredicates:best_is')
        value = Literal(best_env_value)

        kb.add((batchsize_info, best_is, value))
