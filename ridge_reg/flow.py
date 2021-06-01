from metaflow import FlowSpec, Parameter, JSONType, step, S3


class PlayFlow(FlowSpec):
    """
    Game prediction metaflow
    """

    mode = Parameter(
        "mode",
        help="Running mode. 0: backtest, 1: precise prediction, 2: quick prediction",
        default=1,
    )

    @step
    def start(self):
        """parse arguments"""
        print("test flow")
        self.next(self.end)

    @step
    def end(self):
        """end step"""
        pass

if __name__ == "__main__":
    PlayFlow()
