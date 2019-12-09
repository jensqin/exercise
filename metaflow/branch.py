from metaflow import FlowSpec, step


class BranchFlow(FlowSpec):
    """
    Branch Flow example
    """

    @step
    def start(self):
        self.next(self.a, self.b)

    @step
    def a(self):
        self.x = 1
        self.next(self.join)

    @step
    def b(self):
        self.x = 2
        self.next(self.join)

    @step
    def join(self, inputs):
        print(f'a is {inputs.a.x}')
        print(f'b is {inputs.b.x}')
        total = sum(input.x for input in inputs)
        print(f'total is {total}.')
        self.next(self.end)

    @step
    def end(self):
        """
        End step
        """
        print('This is end step.')
        
if __name__ == "__main__":
    BranchFlow()
