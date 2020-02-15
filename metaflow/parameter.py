from metaflow import FlowSpec, Parameter, step


class ParameterFlow(FlowSpec):
    alpha = Parameter("alpha", help="Learning rate", default=0.01)

    # separator is useful for range type parameters
    # only allowed for string parameters
    # examples
    # $ python parameter.py run --phi 1
    # self.phi = ['1']
    # $ python parameter.py run --phi 1:2
    # self.phi = ['1', '2']
    phi = Parameter("phi", default="1:2", separator=":")

    def __init__(self):
        super().__init__()
        self.beta = 0
        self.eta = 0

    @step
    def start(self):
        print(f"alpha is {self.alpha}")
        self.beta = self.alpha + 1
        self.eta = 2
        print(self.phi)
        self.next(self.end)

    @step
    def end(self):
        self.eta = self.eta + 1
        print(f"alpha is now {self.alpha}")


if __name__ == "__main__":
    ParameterFlow()
