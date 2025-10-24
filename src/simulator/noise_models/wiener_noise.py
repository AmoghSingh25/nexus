from jax import random
import jax.numpy as jnp


class WienerNoise:
    def __init__(self, delta=0.01, random_key=42):
        self.delta = delta
        self.delta_sq = jnp.sqrt(self.delta)
        self.key, self.sub_key = random.split(random.key(random_key))

    def generate_noise(self, shape=(1,)):
        noise = self.delta_sq * random.normal(key=self.sub_key, shape=shape)
        self.key, self.sub_key = random.split(self.key)
        return noise
