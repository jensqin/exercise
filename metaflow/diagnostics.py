from metaflow import Flow, get_metadata, Step


print(f"Current metadata provider: {get_metadata()}")
run = Flow('ParameterFlow').latest_successful_run
print(run.id)

# print alpha after the last step
print(run.data.alpha)
print(run.data.eta)

# print alpha at the first step
step1 = Step(f'ParameterFlow/{run.id}/start')
print(step1.task.data.eta)
