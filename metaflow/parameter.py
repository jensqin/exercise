from metaflow import FlowSpec, Parameter, step


class ParameterFlow(FlowSpec):
    alpha = Parameter(
        'alpha', help='Learning rate', default=0.01
    )

    def __init__(self):
        super().__init__()
        self.beta = 0
        self.eta = 0

    @step
    def start(self):
        print(f'alpha is {self.alpha}')
        self.beta = self.alpha + 1
        self.eta = 2
        self.next(self.end)

    @step
    def end(self):
        self.eta  = self.eta  + 1
        print(f'alpha is now {self.alpha}')

if __name__ == "__main__":
    ParameterFlow()
