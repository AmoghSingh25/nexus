from jax import random


def generate_jax_random(func, key, sub_key, req_shape):
    ret = func(key=sub_key, shape=req_shape)
    key, sub_key = random.split(key)
    return key, sub_key, ret
