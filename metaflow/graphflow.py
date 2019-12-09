from metaflow import FlowSpec, step


class MergeFlow(FlowSpec):

    @step
    def start(self):
        self.pass_down = 'a'
        self.next(self.a, self.b)

    @step
    def a(self):
        self.common = 5
        self.x = 1
        self.y = 3
        self.from_a = 6
        self.next(self.join)

    @step
    def b(self):
        self.common = 5
        self.x = 2
        self.y = 4
        self.next(self.join)

    @step
    def join(self, inputs):
        self.x = inputs.a.x
        self.merge_artifacts(inputs, exclude=['y'])
        print(f'x is {self.pass_down}')
        print(f'common is {self.common}')
        print(f'from_a is {self.from_a}')
        self.next(self.end)

    @step
    def end(self):
        print('This is the end step.')

if __name__ == "__main__":
    MergeFlow()
