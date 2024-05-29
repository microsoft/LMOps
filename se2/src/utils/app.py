

class App:
    def __init__(self):
        self.functions = {}
    def add(self, key):
        def adder(func):
            self.functions[key] = func
            return func
        return adder
    def __getitem__(self, __name: str) :
        return self.functions[__name]
