from metaflow import step
from baseflow import BaseFlow


class SubFlow(BaseFlow):
    @step
    def start(self):
        self.next(self.step1)

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
