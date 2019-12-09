from metaflow import FlowSpec, Parameter, step


class ParameterFlow(FlowSpec):
    alpha = Parameter(
        'alpha', help='Learning rate', default=0.01
    )

    @step
    def start(self):
        print(f'alpha is {self.alpha}')
        self.next(self.end)

    @step
    def end(self):
        print(f'alpha is still {self.alpha}')

if __name__ == "__main__":
    ParameterFlow()
