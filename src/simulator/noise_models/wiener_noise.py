from jax import random
import jax.numpy as jnp


class WienerNoise:
    """
    Class for creating a Wiener noise generation object.
    """

    def __init__(self, delta=0.01, random_key=42):
        """
        Initialize Wiener Noise

        :param self: WienerNoise
        :param delta: Simulation time step delta
        :param random_key: Random key for JAX random generation
        """
        self.delta = delta
        self.delta_sq = jnp.sqrt(self.delta)
        self.key, self.sub_key = random.split(random.key(random_key))

    def generate_noise(self, shape=(1,)):
        """
        Generates Wiener noise of the given shape

        :param self: WienerNoise
        :param shape: Required shape of the noise
        """
        noise = self.delta_sq * random.normal(key=self.sub_key, shape=shape)
        self.key, self.sub_key = random.split(self.key)
        return noise
