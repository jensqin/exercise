from metaflow import FlowSpec, step


class BaseFlow(FlowSpec):
    @step
    def start(self):
        print("this is a start")
        self.next(self.step1)

    @step
    def step1(self):
        print("base step 1")
        self.next(self.end)

    @step
    def end(self):
        print("base step end.")


class SubFlow(BaseFlow):
    @step
    def step1(self):
        print("sub step 1")
        self.next(self.step2)

    @step
    def step2(self):
        print("sub step 2")
        self.next(self.end)


if __name__ == "__main__":
    SubFlow()
