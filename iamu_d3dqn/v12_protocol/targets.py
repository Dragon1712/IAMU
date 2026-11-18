"""Pure macro Bellman target, shared by V1.2 agent and regression tests."""
def ddqn_interval_target(interval_reward,bootstrap_discount,next_value):
    return interval_reward+bootstrap_discount*next_value
