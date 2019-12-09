%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from rl_glue import RLGlue
import main_agent
import ten_arm_env
import test_env
from tqdm import tqdm
import time

#agent with the highest expected award with GREEDY
def argmax(q_values):
	#takes a list and returns index of the highest value item->randomly
	top=float("-inf")
	ties=[]
	for i in range(len(q_val)):
		if q_values[i] > top:
			top, ties=q_val[i], [i] #update top, reset ties-0
		elif q_values[i] == top:
			ties.append(i)
	#return random
	#ind-index of the highest val in q_val
	ind=np.random.choice(ties)
	return ind

########################TEST
test_array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
assert argmax(test_array) == 8, "argmax returns the index of the largest value"
test_array = [1, 0, 0, 1]
total = 0
for i in range(100):
    total += argmax(test_array)
np.save("argmax_test", total)
assert total > 0, "randomly choooses largest values && do not use np.random.seed"
assert total != 300, "randomly choooses among the largest values."
########################

# greedy with agent step-> gets called each time the agent takes a step, also it uses the enviroment
class GreedyAgent(main_agent.Agent):
	def agent_step(slef, reward, observation):
		self.arm_count[self.last_action] +=1
		self.q_values[self.last_action]+=(reward - self.q_values[self.last_action])/self.arm_count[self.last_action]
		current_action=argmax(self.q_values)
		self.last_action = current_action
        
        return current_action

#takes:one step, reward and obs and return the action the agent chooses at that time step
#reward from env after last action
#obs: his state
#arm_count->array with the cout of the nr each arm has been pulles
#q_values->agents values estimates for each action
#last_a-> the action that the agent tool on the previous time step

''''
greedy_agent = GreedyAgent()
greedy_agent.q_values = [0, 0, 1.0, 0, 0]
greedy_agent.arm_count = [0, 1, 0, 0, 0]
greedy_agent.last_action = 1
action = greedy_agent.agent_step(1, 0)
print(greedy_agent.q_values)
np.save("greedy_test", greedy_agent.q_values)
print("Output:")
print(greedy_agent.q_values)
print("Expected Output:")
print([0, 0.5, 1.0, 0, 0])
assert action == 2, "using argmax to choose the action with the highest value."
assert greedy_agent.q_values == [0, 0.5, 1.0, 0, 0], "updating q_values correctly."
'''

##################
num_runs=200 
num_steps=1000 #for each exp
env=ten_arm_env.Enviroment
agent=GreedyAgent
agent_info={"num_actions":10} #nr of arms
enf_info={}
all_averages=[]

for i in tqdm(range(num_runs)): #tqdm->progress bar
	rl_glue=RLGlue(env, agent) #creates the experiemt
	rl_glue.rl_init(agent_info, env_info)
	rl_glue.rl_start()
	scores=[0]
	average=[]

	for i in range(num_steps):
		reward, _, action, _=rl_glue.rl_step()#agent and env take a step and return
		scored.append(scores[-1] + reward)
		average.append(scored[-1]/(i+1))
	all_averages.append(averages)

plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.plot([1.55 for _ in range(num_steps)], linestyle="--")
plt.plot(np.mean(all_averages, axis=0))
plt.legend(["Best Possible", "Greedy"])
plt.title("Average Reward of Greedy Agent")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.show()
greedy_scores = np.mean(all_averages, axis=0)
np.save("greedy_scores", greedy_scores)



#EPSILON GREEDY -> introducing a random nr
class EpsilonGreedyAgent(main_agent.Agent):
	def agent_step(self, reward, observation):
		if np.random.random() < self.epsilon:
			current_action=np.random.randint(0, len(self.g_values))
		else:
			current_action = argmax(self.q_values)

		self.arm_count[self.last_action] +=	1
		self.q_values[self.last_action] = self.q_values[self.last_action] + 1/(self.arm_count[self.last_action]) * (reward - self.q_values[self.last_action])

		self.last_action=current_action
		return current_action
'''
e_greedy_agent = EpsilonGreedyAgent()
e_greedy_agent.q_values = [0, 0, 1.0, 0, 0]
e_greedy_agent.arm_count = [0, 1, 0, 0, 0]
e_greedy_agent.num_actions = 5
e_greedy_agent.last_action = 1
e_greedy_agent.epsilon = 0.5
action = e_greedy_agent.agent_step(1, 0)
print("Output:")
print(e_greedy_agent.q_values)
print("Expected Output:")
print([0, 0.5, 1.0, 0, 0])
#assert action == 2, "argmax to choose the action with the highest value."
assert e_greedy_agent.q_values == [0, 0.5, 1.0, 0, 0], "updating q_values correctly."
'''

#DIFFERENT VALUES FOR EPSILON
epsilons = [0.0, 0.01, 0.1, 0.4]

plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.plot([1.55 for _ in range(num_steps)], linestyle="--")

n_q_values = []
n_averages = []
n_best_actions = []

num_runs = 200

for epsilon in epsilons:
    all_averages = []
    for run in tqdm(range(num_runs)):
        agent = EpsilonGreedyAgent
        agent_info = {"num_actions": 10, "epsilon": epsilon}
        env_info = {"random_seed": run}

        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()
        
        best_arm = np.argmax(rl_glue.environment.arms)

        scores = [0]
        averages = []
        best_action_chosen = []
        
        for i in range(num_steps):
            reward, state, action, is_terminal = rl_glue.rl_step()
            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))
            if action == best_arm:
                best_action_chosen.append(1)
            else:
                best_action_chosen.append(0)
            if epsilon == 0.1 and run == 0:
                n_q_values.append(np.copy(rl_glue.agent.q_values))
        if epsilon == 0.1:
            n_averages.append(averages)
            n_best_actions.append(best_action_chosen)
        all_averages.append(averages)
        
    plt.plot(np.mean(all_averages, axis=0))
plt.legend(["Best Possible"] + epsilons)
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.show()


#STEP SIZE
class EpsilonGreedyAgentConstantStepsize(main_agent.Agent):
    def agent_step(self, reward, observation):
        if np.random.random() < self.epsilon:
            current_action = np.random.randint(0, len(self.q_values))
        else:
            current_action = argmax(self.q_values)
        self.q_values[self.last_action] = self.q_values[self.last_action] + self.step_size * (reward - self.q_values[self.last_action])
        self.last_action = current_action
        return current_action


step_sizes = [0.01, 0.1, 0.5, 1.0]
epsilon = 0.1
num_steps = 1000
num_runs = 200
fig, ax = plt.subplots(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
q_values = {step_size: [] for step_size in step_sizes}
true_values = {step_size: None for step_size in step_sizes}
best_actions = {step_size: [] for step_size in step_sizes}

for step_size in step_sizes:
    all_averages = []
    for run in tqdm(range(num_runs)):
        agent = EpsilonGreedyAgentConstantStepsize
        agent_info = {"num_actions": 10, "epsilon": epsilon, "step_size": step_size, "initial_value": 0.0}
        env_info = {"random_seed": run}

        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()
        
        best_arm = np.argmax(rl_glue.environment.arms)

        scores = [0]
        averages = []
        
        if run == 0:
            true_values[step_size] = np.copy(rl_glue.environment.arms)
            
        best_action_chosen = []
        for i in range(num_steps):
            reward, state, action, is_terminal = rl_glue.rl_step()
            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))
            if action == best_arm:
                best_action_chosen.append(1)
            else:
                best_action_chosen.append(0)
            if run == 0:
                q_values[step_size].append(np.copy(rl_glue.agent.q_values))
        best_actions[step_size].append(best_action_chosen)
    ax.plot(np.mean(best_actions[step_size], axis=0))
    if step_size == 0.01:
        np.save("step_size", best_actions[step_size])
    
ax.plot(np.mean(n_best_actions, axis=0))
fig.legend(step_sizes + ["1/N(A)"])
plt.title("% Best Action Taken")
plt.xlabel("Steps")
plt.ylabel("% Best Action Taken")
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
plt.show()

#each
largest = 0
num_steps = 1000
for step_size in step_sizes:
    plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
    largest = np.argmax(true_values[step_size])
    plt.plot([true_values[step_size][largest] for _ in range(num_steps)], linestyle="--")
    plt.title("Step Size: {}".format(step_size))
    plt.plot(np.array(q_values[step_size])[:, largest])
    plt.legend(["True Expected Value", "Estimated Value"])
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.show()

plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.title("Step Size: 1/N(A)")
plt.plot([true_values[step_size][largest] for _ in range(num_steps)], linestyle="--")
plt.plot(np.array(n_q_values)[:, largest])
plt.legend(["True Expected Value", "Estimated Value"])
plt.xlabel("Steps")
plt.ylabel("Value")
plt.show()