from jax import random


class Chemical:
    def __init__(self, key=42, name="chem1"):
        self.key, self.sub_key = random.split(random.key(key))
        self.mass = random.uniform(key=self.sub_key, shape=(1))
        self.key, self.sub_key = random.split(self.key)
        self.name = name

    def __repr__(self):
        return f"{self.name} \n Mass - {self.mass}\n"
