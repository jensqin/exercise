from metaflow import FlowSpec, step


class ForeachFlow(FlowSpec):
    """
    Foreach Flow
    """

    @step
    def start(self):
        self.titles = [
            'football', 'basketball', 'soccer'
        ]
        self.next(self.a, foreach='titles')

    @step
    def a(self):
        self.title = f'{self.input} processed'
        self.next(self.b)

    @step
    def b(self):
        self.title += ' done'
        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = tuple(input.title for input in inputs)
        self.next(self.end)

    @step
    def end(self):
        print('\n'.join(self.results))

if __name__ == "__main__":
    ForeachFlow()
