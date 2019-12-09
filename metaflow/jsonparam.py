from metaflow import FlowSpec, Parameter, step, JSONType


class JSONParameterFlow(FlowSpec):
    gdp = Parameter(
        'gdp', help='country-gdp mapping', 
        type=JSONType, default='{"US": 1939}'
    )
    country = Parameter(
        'country', help='choose a country',
        default='US'
    )

    @step
    def start(self):
        print(f'The GDP of {self.country} is ${self.gdp[self.country]}')
        self.next(self.end)

    @step
    def end(self):
        print('This is the end step.')

if __name__ == "__main__":
    JSONParameterFlow()
