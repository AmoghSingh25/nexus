from jax import random

"""
Manage the random generation function of JAX, performs key split and returns new key and subkey
"""


def generate_poisson(key, sub_key, **kwargs):
    ret = random.poisson(key=sub_key, **kwargs)
    key, sub_key = random.split(key)
    return key, sub_key, ret


def generate_uniform(key, sub_key, **kwargs):
    ret = random.uniform(key=sub_key, **kwargs)
    key, sub_key = random.split(key)
    return key, sub_key, ret


def generate_permutation(key, sub_key, **kwargs):
    ret = random.permutation(key=sub_key, **kwargs)
    key, sub_key = random.split(key)
    return key, sub_key, ret


def generate_choices(key, sub_key, **kwargs):
    ret = random.choice(key=sub_key, **kwargs)
    key, sub_key = random.split(key)
    return key, sub_key, ret


def generate_normal(key, sub_key, mean=0, **kwargs):
    ret = random.normal(key=sub_key, **kwargs)
    key, sub_key = random.split(key)
    return key, sub_key, mean + ret
