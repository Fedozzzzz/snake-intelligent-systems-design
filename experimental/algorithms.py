import rdflib


def split_res(string):
    return string.split(':')[-1]


def call_algorithms(alg_name, obj={}, res=[], additional=[]):
    print('')
