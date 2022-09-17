import json

from metaflow import FlowSpec, step, IncludeFile, S3


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
        with S3(s3root="s3://bla-basketball-models/examples") as s3:
            s3.put("example.json", self.params)
        print(type(self.params))
        print(json.loads(self.params))
        print("hello world")
        self.next(self.end)

    @step
    def end(self):
        """
        sample end
        """
        with S3(s3root="s3://bla-basketball-models/examples") as s3:
            example = s3.get("example.json")
            print(example.text)
            dict_text = json.loads(example.text)
            print(type(dict_text))  # dict
        print("end.")


if __name__ == "__main__":
    HelloFlow()
