

<div align="center">
<img src="./assets/logo.png" style="width: 20%;height: 10%">
<h1>
  <span style="color: rgb(0, 176, 80);">D</span><span style="color: rgb(250, 196, 34);">S</span><span style="color: rgb(0, 176, 240);">G</span>Bench: A 
  <span style="color: rgb(0, 176, 80);">D</span>iverse <span style="color: rgb(250, 196, 34);">S</span>trategic <span style="color: rgb(0, 176, 240);">G</span>ame Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments

</h1>

</div>

<p align="center">
  <i>"The journey of strategic games is endless, where agents evolve to master the art of decision-making. This release represents our first step towards that vision."</i>
</p>


<p align="center">
  📄 <a href="https://arxiv.org/abs/2503.06047" target="_blank">Paper</a> • 🌍 <a href="https://fateyetian.github.io/DSGBench/" target="_blank">Project Page</a> • 🎮 <a href="https://decibrain-group.github.io/GameWiki/" target="_blank">Game Wiki</a> • 💻 <a href="https://github.com/DeciBrain-Group/DSGBench" target="_blank">Code</a><br>
</p>

## 🔥 News & Updates

- **[2026.05]** 🚀 **DSGBench++** is on the way! *A Closed-Loop Infrastructure for Agentic Strategic Decision-Making* — integrating Evaluation, Trajectory Collection, and RL Training into a unified agentic loop. *"In the agentic era, true intelligence is forged not in isolated tests, but in the continuous loop of decision, feedback, and growth."*

## 📖 Introduction

DSGBench is a novel strategic game benchmark designed to evaluate the performance of LLM-based agents in strategic planning, real-time decision making, adaptability, and multi-agent interactions. The benchmark encompasses six high-dynamics, complex strategy games, including *StarCraft II*, *Civilization*, *Street Fighter III*, and others. It provides fine-grained metrics for comprehensive evaluation, along with detailed decision trajectory analysis and trajectory datasets.

![DSGBench Overview](./assets/overview.png)

## 🌟 Highlights

- **Strategic Games**: DSGBench fills an important gap in evaluating agents in complex strategic environments. It reflects the agent's overall capabilities in real-world applications, beyond just individual skill tests.
  
- **Scenario Diversity**: DSGBench offers a variety of scenario settings, allowing comprehensive testing of agent adaptability and generalization. An ideal agent should excel in diverse environments and demonstrate cross-scenario generalization rather than optimizing for a single task.

- **Fine-grained Evaluation Metrics and Decision Trajectory Analysis**: DSGBench employs detailed evaluation metrics that cover strategic planning, real-time decision-making, adaptability, and multi-agent interactions. These fine-grained metrics provide deeper quantitative analysis of agent performance, offering insights into strengths and weaknesses across different dimensions, as opposed to relying solely on traditional win-rate metrics.

<!-- - **Trajectory Dataset（To Do）**: DSGBench provides a comprehensive decision trajectory dataset, which serves as a solid data foundation for trajectory fine-tuning (Trajectory SFT) and reinforcement learning from agent feedback (RL from Agent Feedback). -->

## 🌍 Environment Overview

### Game Environments

The evaluation environment for this project consisted of six dynamic, complex strategy games.

| Environment         | Imperfect <br>Info | Strategic & <br>Tactical |Dynamic <br>space| Real-time v.s.<br>Turn-based | More Info                                                      |
|--------------------|----------------|-------------------------|------------|--------------------------|----------------------------------------------------------------|
| StarCraft II       | ✔              | ✔                       |✔         | Real-time                | [scene setting,prompt](./games/starcraft2/readme.md)           |
| Civilization       | ✔              | ✔                       |✔         | Turn-based               | [scene setting,prompt](./games/civ/readme.md)                  |
| Street Fighter III | ✘              | ✘                       |✘         | Real-time                | [scene setting,prompt](./games/streetfight3/readme.md)         |
| Diplomacy          | ✘              | ✔                       |✘         | Turn-based               | [scene setting,prompt](./games/welfare_diplomacy/readme.md)    |
| Werewolf           | ✔              | ✔                       |✘         | Turn-based               | [scene setting,prompt,</br>visual](./games/werewolf/readme.md) |
| Stratego           | ✔              | ✔                       |✘         | Turn-based               | [scene setting,prompt](./games/stratego/readme.md)             |

### Repository Structure
[`games`](./games): This is the game environment directory, which contains 6 games, including StarCraft II, Civilization, Street Fighter III, etc.
Each game uses the OpenAI Gym interface,Its main functions are as follows:
```python
step(action): Updates the environment with an action, returns the next agent observation, the reward for taking that action, whether the environment terminated or truncated due to the latest action, and information about the step in the environment, i.e. metrics and debug information.
reset(): Resets the environment to its initial state, needed before calling step. Returns the first agent observation of an event and information.
render(): Renders the environment to help visualize what the agent sees, example modes are "human", "rgb_array", "ansi" (for text).
close(): Closes the game environment
```
[`agent_manager`](./agent_manager): This is the agent management folder, which implements the agent, prompt, and LLM interfaces of various games.

[`configs`](./configs): This is the configuration folder, used to configure the evaluation game scene parameters, LLM parameters

[`agent_eval`](./agent_eval): This is the evaluation folder, the evaluation indicators are calculated.

[`utils`](utils): This folder is for common functions and log functions

[`create_yaml.py`](create_yaml.py): This script generates specific configurations for multiple LLMs based on the basic configuration under [`configs/eval_config_base`](./configs/eval_config_base).

[`multiprocess_eval_tasks.py`](multiprocess_eval_tasks.py): the running script for the evaluation




## 🚀 Quick Start
### Table of Contents
- [🛠️ Step 1: Prerequisites](#step-1-prerequisites)
- [🎮 Step 2: Game Setup](#step-2-game-setup)
- [🔧 Step 3: Configuration](#step-3-configuration)
- [🏃‍♂️ Step 4: Configure & Run Tasks](#step-4-configure-and-run-tasks)
- [🏃 Step 5: Run Evaluation Tasks](#step-5-run-evaluation-tasks)

### Step 1: Prerequisites

- `Operating System`: Windows 11 with WSL, the reason for using windows is that BLZ didn't release the latest sc2 on liunx, WSL is used to run Civilization and Street Fighter III.
- `Docker management`: Docker Desktop, it provides a simple and intuitive way to install and use Docker on Windows, allowing developers and operators to easily build, test and run Docker containers.
- `python`: python 3.9.6
Dependencies can be installed by running 

```shell
# create virtual environment
conda create --name dsgbench python=3.9.6 
conda activate dsgbench
cd DSGBench

# install PromptCoder
git clone https://github.com/dhh1995/PromptCoder
cd PromptCoder
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..

# install other pypi 
# note: if install slowly, you can user Tsinghua Source (`-i https://pypi.tuna.tsinghua.edu.cn/simple `) to speed up the installation
pip install -r pip_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# reinstall websocket,websocket-client
pip uninstall websocket
pip uninstall websocket-client
pip install websocket==0.2.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install websocket-client==1.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


```

### Step 2: Game Setup
This section details the setup for the required games and game environments.


#### **👉 StarCraft II**

**1.Install StarCraft II**
StatCraft II is a classic game developed by BLZ. You can download Battle.net from [blizzard app](https://www.blizzard.com/zh-tw/)

If you are Chinese, due to the Bobby Kotick, CN play cant own their sever again. So we must download StarCraft II by this video :[video](https://www.bilibili.com/video/BV1As4y147NP/?buvid=XY6E01868C47C929FEFCE4A6DBF0A4ECFFB64&is_story_h5=false&mid=y0%2Bkb3rZVEwQ9j34NFXkLA%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=android&share_plat=android&share_session_id=2e6181cb-fa27-4ce1-9b2e-f126f39267d5&share_source=COPY&share_tag=s_i&timestamp=1674580642&unique_k=rPeGgmE&up_id=149681985&vd_source=0553fe84b5ad759606360b9f2e687a01)
or you can search in the internet.

**2.Maps**
you should put maps Ancient Cistern LE.SC2Map (games/starcraft2/Maps/Ancient Cistern LE.SC2Map) to your StarCrafrt2 file  in StarCraft II\Maps(If the 'Maps' file dont exist, please create it).

#### **👉 Civilization**
Civ requires additional docker image as game engine
```shell
# 1. civ game engine
docker pull akon123456/freeciv-web
docker tag akon123456/freeciv-web:latest freeciv/freeciv-web:latest
# Start the freeciv-web service in conda environment:
conda activate dsgbench
start_freeciv_web_service
# if you need stop freeciv-web service
stop_freeciv_web_service
```
#### **👉 Street Fighter III**

Street Fighter III is a classic fighting game series developed by CAPCOM,in this project, we adopt [DIAMBRA Arena](https://github.com/diambra/) as our game environments.
```shell
# get the DIAMBRA environment
docker pull diambra/engine:v2.2
```
Download this rom [sfiii3n.zip](https://wowroms.com/en/roms/mame/street-fighter-iii-3rd-strike-fight-for-the-futur-japan-clone/106255.html) manually and place it in the [`./games/streetfight3/roms`](./games/streetfight3/roms/) directory

#### **👉 Diplomacy**

Diplomacy：For running experiments with exploiter agents, you must first download the necessary weights.
manually download [fppi2_params.npz (link begins download)](https://storage.googleapis.com/dm-diplomacy/fppi2_params.npz) and place it in the [`./games/welfare_diplomacy/welfare_diplomacy_baselines/network_parameters`](./games/welfare_diplomacy/welfare_diplomacy_baselines/network_parameters/) directory.

#### **👉 Werewolf and Stratego**
These two games do not require any additional setup or configuration.


###  Step 3: Configuration 
#### 3.1 LLM API key
LLM config are stored in the path "configs/llm_configs/"
this project currently supports API calling, so you need configure LLM API key first.
```shell
model_name: *****           # specific model name, like "gpt-3.5-turbo-0125"
api_key: sk-*****           # API key
api_url: https://****.com   # LLM API URL
```

You can configure additional parameters such as `max_tokens`, `timeout`, and `temperature` for fine-tuning your model's performance.
Supported Models:
- `llama-3.1-70b`
- `deepseek-v25`
- `gemini-1.5-flash`
- `gpt-3.5`
- `gpt-4o`
- `gpt-4o-mini`
- `o1-mini`

#### 3.2 Wandb key
We use W&B's tools to quickly track experiments and visualize results. This feature is useful for tracking large model output trajectories, but it is **optional** and can be configured based on your needs.

To configure your W&B key, modify the `tasks_config.py` file:
```python
WANDB_API_KEY = ****your_wandb_key****
```

### Step 4: Configure & Run Tasks

#### Task Configuration Example

You can configure the tasks to be evaluated in the `tasks_config.py` file. For instance, if you want to evaluate a specific task, add the task and its corresponding configuration file:

```yaml
"eval_deepseek-v25_stratego_scene1_deepseek-v25_vs_gpt-4o-mini_random": "eval_deepseek-v25_stratego_scene1_deepseek-v25_vs_gpt-4o-mini_random.yaml"
```
This task will evaluate a game between two agents: a StrategoAgent and a GPT-4o-mini agent. The configuration for this task includes settings for agent models, the number of matches to be played, the output path, and the game settings.

```yaml
agent:
- agent_model: LLMModel
  agent_model_config: deepseek-v25.yaml
  agent_model_config_opp: gpt-4o-mini.yaml
  agent_model_opp: LLMModel
  agent_name: StrategoAgent
  agent_prompt: StategoPrompt
  agent_prompt_opp: StategoPrompt
eval:
  num_matches: 10
  output_path: ./output/deepseek-v25/stratego/scene1/deepseek-v25_vs_gpt-4o-mini_random
  weave_prj_name: eval_deepseek-v25_stratego_scene1_deepseek-v25_vs_gpt-4o-mini_random
game:
  game_init: random
  game_mode: agent_vs_agent
  game_name: StrategoMultiAgentEnv
```
**Explanation of Variables:**

- **agent**: Defines the two agents involved in the task. The `agent_model` and `agent_model_opp` represent the models of the two agents. Each agent has a corresponding prompt file (e.g., `StrategoPrompt`) that provides the context for the agent's behavior.

- **eval**: Specifies the number of matches (`num_matches`) to be played, where the results will be stored (`output_path`), and the name of the task (`weave_prj_name`) for tracking the evaluation.

- **game**: Specifies the type of game to be used for evaluation. The game is initialized in a random state (`game_init: random`), and the mode is set to `agent_vs_agent`, meaning two agents will compete against each other.



> **Note:** Some games, such as **StarCraft II** and **Stratego**, may require more time for evaluation. Please adjust your task selection based on the time you have available.


**Run tasks**
Once your tasks are configured, you can start running them with the following command:

```shell
diambra run --images.no-pull -r [absolute roms path of street fight III] python multiprocess_eval_tasks.py
# example
# diambra run --images.no-pull -r D:\ubuntu\jinxin_work\Agent_Eval\llm-colosseum-main\llm-colosseum-main\roms python multiprocess_eval_tasks.py
```
>**Note:** This command executes the evaluation tasks with the necessary ROMs for the game **Street Fighter III**. Make sure to replace `[absolute roms path of street fight III]` with the actual path to your ROMs directory.  
>  
> The `run` command will execute **all** the tasks configured in the `tasks_config.py` file, not just a single task. The reason the command starts with `diambra run` is due to compatibility with **Street Fighter III**, which requires this command structure. Make sure to configure your tasks properly before running.


### Step 5: Run Evaluation Tasks
```shell
# After opening CMD window
activate dsgbench
cd C:\deploy\DSGBench

# Run tasks
python multiprocess_eval_tasks.py
```

## 🚀 Start with docker compose
### Step 1: Prerequisites
Download the DSGBench disk file [download link](https://pan.baidu.com/s/1hW9VR23CKYOoZy4ahsoHSQ?pwd=1ii7) manually, and cat the data.img_* to data.img file
> **Note**: The image size is over 60GB. Due to game environment compatibility issues, Linux image packaging is not currently supported. We are working to fix this and will provide Linux support as soon as possible.

### Step 2: Directory Structure
```shell
|--dsgbench
    |--data                      #  the win docker disk files
        |--data.img                      #  the windows system disk files
        |--windows.*                      #  the windows system disk files
    |--docker-compose.yml        #  the docker compose files
    |--.diambra                  #  the game files of Street Fighter III
```
### Step 3: Prepare Docker Environment
```shell
cd dsgbench

# Get the Civilization game environment
docker pull akon123456/freeciv-web
docker tag akon123456/freeciv-web:latest freeciv/freeciv-web:latest

# Get the DIAMBRA environment
docker pull diambra/engine:v2.2

# Get the Windows environment
docker pull dockurr/windows:4.14
```

### Step 4: Create Network and Launch
```shell
# Create DSG network
docker network create --subnet=172.22.0.0/24 dsg_network

# Launch Docker Compose
docker compose up -d

# Access via browser: http://localhost:8006
```

### Step 5: Run Evaluation Tasks
```shell
# After opening CMD window
activate dsgbench
cd C:\deploy\DSGBench

# Run tasks
python multiprocess_eval_tasks.py
```

## 🗂️ Result Structure
The results include two main types: **wandb trace** and **txt result**.
#### wandb result
Each time you run the task, the **wandb project address** will be output. **Wandb** mainly records the specific indicator trends of the agent's interaction with the environment.
#### txt result
```shell
|--ability_score
    |--model_ability_scores.csv       #  the final score of the model
    |--game_ability_scores.csv        #  the score of the model on each game
    |--calculation_steps.json         #  the score process
    |--ability_detail_score           #  the model's scores in different game scenarios
        |--civ.csv                    
        |--starcraft2.csv    
        |--stratego.csv    
        |--streetfight3.csv    
        |--welfare_diplomacy.csv    
        |--werewolf.csv    
|--[model_name]
    |--[game_name]
        |--[scene_name]
            |--[diff_opp]
                |--*.log              # run log, Contains the input and output of each LLM call 
                |--*.json             # match trajectory 
                |--*.csv              # result for per match
             

```
> **Note:** The structure represents a hierarchical view of the results, with ability_score as the main folder that includes model and game-specific performance data. Each game or scene has its detailed logs and results, separated by opponent types and scenarios.

## 🔮 Future Work
Strategic games represent an infinite arena where LLM-based agents can truly demonstrate their potential.  Through the dynamic interactions with opponents, agents not only learn and adapt but also develop increasingly complex strategies - much like how humans master the art of strategy. While this benchmark marks an important first step, there are several promising directions we envision for future exploration:

1. **Creating a Unified Dataset for Strategic Game Scenarios**  
   To perform in-depth strategic game evaluations, we will build a **standardized trajectory dataset** that covers various strategic scenarios. These datasets will provide rich training resources for trajectory-based supervised fine-tuning (SFT) and reinforcement learning (RL) feedback, contributing to more robust model development.

2. **Developing a Hierarchical Evaluation Framework**  
   Evaluation should not be a one-time check but rather an ongoing process. We plan to introduce a **gradient hierarchical evaluation method**, which gradually increases task complexity from basic tasks to more intricate scenarios. This progressive approach will enable a deeper understanding of how models adapt and evolve in diverse situations.

3. **Implementing the Elo Rating System**  
   The competitive nature of the games requires the integration of the **Elo rating system**, a dynamic ranking mechanism that more accurately reflects the relative abilities of models. This system helps avoid the biases introduced by static scoring and provides a more realistic competitive environment for model evaluation.

4. **Developing a Unified Reinforcement Learning Framework for LLM-based Agent**  
   We aim to develop a unified reinforcement learning framework that integrates opponent modeling and strategic decision-making, enabling agents to better understand opponents and adapt their strategies in complex game scenarios, continuously enhancing the capabilities of LLM-based agents.


## 🔄 Extending DSGBench
The extension specifications for DSGBench will be released soon, allowing the community to contribute new game environments and evaluation scenarios. Stay tuned!


## 📄 License
The DSGBench codebase is licensed under a [MIT License](https://opensource.org/licenses/MIT).
- **TextStarCraft2**: [MIT License](https://opensource.org/licenses/MIT)
- **CivRealm**: [GNU General Public License v3 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html)
- **welfare-diplomacy**: [GNU AFFERO GENERAL PUBLIC LICENSE](https://www.gnu.org/licenses/agpl-3.0.html)
- **Stratego Env**: [MIT License](https://opensource.org/licenses/MIT)
- **llm-colosseum/Street Fighter**: [MIT License](https://opensource.org/licenses/MIT)
- **werewolf_arena**: [Apache License](https://www.apache.org/licenses/LICENSE-2.0)


## 🤝 Acknowledgement
This repo benefits from [TextStarCraft2](https://github.com/histmeisah/Large-Language-Models-play-StarCraftII/), [CivRealm](https://github.com/bigai-ai/civrealm), [welfare-diplomacy](https://github.com/mukobi/welfare-diplomacy), [Stratego Env](https://github.com/JBLanier/stratego_env), [llm-colosseum](https://github.com/OpenGenerativeAI/llm-colosseum) and [werewolf_arena](https://github.com/google/werewolf_arena). Thanks for their wonderful works.

## 📧 Contact
For questions and discussions, please contact: wenjietang2022@163.com



## 📚 Citation
```bibtex
@misc{tang2025dsgbenchdiversestrategicgame,
      title={DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments}, 
      author={Wenjie Tang and Yuan Zhou and Erqiang Xu and Keyan Cheng and Minne Li and Liquan Xiao},
      year={2025},
      eprint={2503.06047},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.06047}, 
}
```


