from metaflow import FlowSpec, step


class HelloFlow(FlowSpec):
    """
    hello world
    install metaflow beforehand
    """

    @step
    def start(self):
        """
        sample start
        """
        print('starting.')
        self.next(self.hello)

    @step
    def hello(self):
        """
        hello world function
        """
        print('hello world')
        self.next(self.end)

    @step
    def end(self):
        """
        sample end
        """
        print('end.')

if __name__ == "__main__":
    HelloFlow()
