import json

from metaflow import FlowSpec, step, IncludeFile


class HelloFlow(FlowSpec):
    """
    hello world
    install metaflow beforehand
    """

    params = IncludeFile("example", default="example.json")

    @step
    def start(self):
        """
        sample start
        """
        print("starting.")
        self.next(self.hello)

    @step
    def hello(self):
        """
        hello world function
        """
        print(type(self.params))
        print(json.loads(self.params))
        print("hello world")
        self.next(self.end)

    @step
    def end(self):
        """
        sample end
        """
        print("end.")


if __name__ == "__main__":
    HelloFlow()
