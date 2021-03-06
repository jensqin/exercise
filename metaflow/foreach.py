from metaflow import FlowSpec, step, batch


class ForeachFlow(FlowSpec):
    """
    Foreach Flow
    """

    @step
    def start(self):
        self.titles = ["football", "basketball", "soccer"]
        self.next(self.a, foreach="titles")

    @batch(cpu=1, memory=4000)
    @step
    def a(self):
        self.title = f"{self.input} processed"
        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = tuple(input.title for input in inputs)
        self.next(self.end)

    @step
    def end(self):
        print("\n".join(self.results))


if __name__ == "__main__":
    ForeachFlow()
