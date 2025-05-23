import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Literal, Optional

import numpy as np
from gymnasium import spaces
from loguru import logger
from agent_manager.prompts import StreetFight3Prompt

from .config import (
    INDEX_TO_MOVE,
    META_INSTRUCTIONS,
    META_INSTRUCTIONS_WITH_LOWER,
    MOVES,
    NB_FRAME_WAIT,
    X_SIZE,
    Y_SIZE,
)
from .observer import detect_position_from_color

from agent_manager.agents.trajectory import Trajectory,set_action_info,set_state_info,set_reward
class Robot:
    observations: List[Optional[dict]] = None  # memory
    next_steps: List[int]  # action plan
    actions: dict  # actions of the agents during a step of the game
    # actions of the agents during the previous step of the game
    previous_actions: Dict[str, List[int]]
    reward: float  # reward of the agent

    action_space: spaces.Space
    character: Optional[str] = None  # character name
    side: int  # side of the stage where playing: 0 = left, 1 = right
    current_direction: Literal["Left", "Right"]  # current direction facing
    sleepy: Optional[bool] = False  # if the robot is sleepy
    only_punch: Optional[bool] = False  # if the robot only punch

    model: str  # model of the robot
    super_bar_own: int
    player_nb: int  # player number

    def __init__(
        self,
        action_space: spaces.Space,
        character: str,
        side: int,
        character_color: list,
        ennemy_color: list,
        sleepy: bool = False,
        only_punch: bool = False,
        model: str = "",
        prompt_templete:StreetFight3Prompt=None,
        player_nb: int = 0,  # 0 means not specified
        weave_prj_name="StreetFight3",
        logger=None
    ):
        self.action_space = action_space
        self.character = character
        if side == 0:
            self.current_direction = "Right"
        elif side == 1:
            self.current_direction = "Left"

        self.observations = []
        self.next_steps = []
        self.character_color = character_color
        self.ennemy_color = ennemy_color
        self.side = side
        self.sleepy = sleepy
        self.only_punch = only_punch
        self.model = model
        self.prompt_templete = prompt_templete
        self.previous_actions = defaultdict(list)
        self.actions = {}
        self.player_nb = player_nb
        self.weave_prj_name=weave_prj_name
        self.generate_times = 1
        self.grounding_errors = 0
        self.next_skills=[]
        self.logger=logger
        ## trajectory
        self.trajectory: Trajectory = []
        self.cur_time_step = 0
        self.role="player" if player_nb==1 else "opp_player"

    def act(self) -> int:
        """
        At each game frame, we execute the first action in the list of next steps.

        An action is an integer from 0 to 18, where 0 is no action.

        See the MOVES dictionary for the mapping of actions to moves.
        """
        if not self.next_steps or len(self.next_steps) == 0:
            return 0  # No move

        if self.sleepy:
            return 0

        if self.only_punch:
            # Do a Hadouken
            if self.current_direction == "Right":
                self.next_steps.extend(
                    [
                        MOVES["Down"],
                        MOVES["Right+Down"],
                        MOVES["Right"],
                        MOVES["High Punch"],
                    ]
                )
            elif self.current_direction == "Left":
                self.next_steps.extend(
                    [
                        MOVES["Down"],
                        MOVES["Down+Left"],
                        MOVES["Left"],
                        MOVES["High Punch"],
                    ]
                )

        next_step = self.next_steps.pop(0)

        return next_step

    def plan(self) -> None:
        """
        The robot will plan its next steps by calling this method.

        In SF3, moves are based on combos, which are list of actions that must be executed in a sequence.

        Moves of Ken
        https://www.eventhubs.com/guides/2009/may/11/ken-street-fighter-3-third-strike-character-guide/

        Moves of Ryu
        https://www.eventhubs.com/guides/2008/may/09/ryu-street-fighter-3-third-strike-character-guide/
        """

        # If we already have a next step, we don't need to plan
        if len(self.next_steps) > 0:
            return

        # Call the LLM to get the next steps
        next_steps_from_llm = self.get_moves_from_llm()

        print("---------------")
        print("next_buttons_to_press: ",next_steps_from_llm)
        print("---------------")
        next_buttons_to_press = [
            button
            for combo in next_steps_from_llm
            for button in META_INSTRUCTIONS_WITH_LOWER[combo][
                self.current_direction.lower()
            ]
            # We add a wait time after each button press
            + [0] * NB_FRAME_WAIT
        ]


        if len(self.next_skills)>=5:
            self.next_skills=self.next_skills[1:]
        if len(self.next_skills)<=5:
            self.next_skills.extend([{combo: [button for button in META_INSTRUCTIONS_WITH_LOWER[combo][self.current_direction.lower()]]}
                for combo in next_steps_from_llm])
        for skill in self.next_skills:
            for com,bt in skill.items():
                while len(bt):
                    if bt[-1]==0:
                        bt=bt[:-1]
                    else:
                        break
        print("next_buttons_to_press: ", self.next_skills)
        print("---------------")
        self.next_steps.extend(next_buttons_to_press)

    def observe(self, observation: dict, actions: dict, reward: float):
        """
        The robot will observe the environment by calling this method.

        The latest observations are at the end of the list.
        """

        # detect the position of characters and ennemy based on color
        observation["character_position"] = detect_position_from_color(
            observation, self.character_color
        )
        observation["ennemy_position"] = detect_position_from_color(
            observation, self.ennemy_color
        )
        self.cur_time_step = observation.get("_time_step",1)
        self.observations.append(observation)
        # we delete the oldest observation if we have more than 10 observations
        if len(self.observations) > 10:
            self.observations.pop(0)

        self.reward = reward

        if actions.get("agent_0") is not None and actions.get("agent_0") != 0:
            self.previous_actions["agent_0"].append(actions["agent_0"])
        if actions.get("agent_1") is not None and actions.get("agent_1") != 0:
            self.previous_actions["agent_1"].append(actions["agent_1"])

        for key, value in actions.items():
            if len(self.previous_actions[key]) > 10:
                self.previous_actions[key].pop(0)

        # Keep track of the current direction by checking the position of the character
        # and the ennemy
        character_position = observation.get("character_position")
        ennemy_position = observation.get("ennemy_position")
        if (
            character_position is not None
            and ennemy_position is not None
            and len(character_position) == 2
            and len(ennemy_position) == 2
        ):
            if character_position[0] < ennemy_position[0]:
                self.current_direction = "Right"
            else:
                self.current_direction = "Left"

    def context_prompt(self) -> str:
        """
        Return a str of the context

        "The observation for you is Left"
        "The observation for the opponent is Left+Up"
        "The action history is Up"
        """

        # Create the position prompt
        side = self.side
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("self.observations[-1]",self.observations[-1].keys())
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("self.observations[-1]",self.observations[-1])
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        obs_own = self.observations[-1]["character_position"]
        obs_opp = self.observations[-1]["ennemy_position"]
        super_bar_own = self.observations[-1]["P" + str(side + 1)]["super_bar"][0]

        if obs_own is not None and obs_opp is not None:
            relative_position = np.array(obs_own) - np.array(obs_opp)
            normalized_relative_position = [
                relative_position[0] / X_SIZE,
                relative_position[1] / Y_SIZE,
            ]
        else:
            normalized_relative_position = [0.3, 0]

        opp_side=1 if side==0 else 0
        health=int(self.observations[-1]["P" + str(side + 1)]['health'])
        opp_health=int(self.observations[-1]["P" + str(opp_side + 1)]['health'])
        # print(f"====={side}")
        health_prompt="Your current health  is {}, and ennemy current health is {}.".format(health,opp_health)
        # print(f"====={health_prompt}")
        position_prompt = ""
        if abs(normalized_relative_position[0]) > 0.1:
            position_prompt += (
                "You are very far from the opponent. Move closer to the opponent."
            )
            if normalized_relative_position[0] < 0:
                position_prompt += "Your opponent is on the right."
            else:
                position_prompt += "Your opponent is on the left."

        else:
            position_prompt += "You are close to the opponent. You should attack him."

        power_prompt = ""
        if super_bar_own >= 30:
            power_prompt = "You can now use a powerfull move. The names of the powerful moves are: Megafireball, Super attack 2."
        if super_bar_own >= 120 or super_bar_own == 0:
            power_prompt = "You can now only use very powerfull moves. The names of the very powerful moves are: Super attack 3, Super attack 4"
        # Create the last action prompt
        last_action_prompt = ""
        if len(self.previous_actions.keys()) >= 0:
            act_own_list = self.previous_actions["agent_" + str(side)]
            act_opp_list = self.previous_actions["agent_" + str(abs(1 - side))]

            if len(act_own_list) == 0:
                act_own = 0
            else:
                act_own = act_own_list[-1]
            if len(act_opp_list) == 0:
                act_opp = 0
            else:
                act_opp = act_opp_list[-1]

            str_act_own = INDEX_TO_MOVE[act_own]
            str_act_opp = INDEX_TO_MOVE[act_opp]

            last_action_prompt += f"Your last action was {str_act_own}. The opponent's last action was {str_act_opp}."

        reward = self.reward

        # Create the score prompt
        score_prompt = ""
        if reward > 0:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif reward < 0:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )

        # Assemble everything
        context = f"""{position_prompt}
{health_prompt}
{power_prompt}
{last_action_prompt}
Your current score is {reward}. {score_prompt}
To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
"""

        return context

    def get_moves_from_llm(
        self,
    ) -> List[str]:
        """
        Get a list of moves from the language model.
        """

        # Filter the moves that are not in the list of moves
        invalid_moves = []
        valid_moves = []

        # If we are in the test environment, we don't want to call the LLM
        if os.getenv("DISABLE_LLM", "False") == "True":
            # Choose a random int from the list of moves
            logger.debug("DISABLE_LLM is True, returning a random move")
            return [random.choice(list(MOVES.values()))]

        while len(valid_moves) == 0:
            llm_stream = self.call_llm()
            self.generate_times += 1
            # adding support for streaming the response
            # this should make the players faster!
            llm_response = ""
            for r in [llm_stream]:
                # print(r.delta, end="")
                llm_response += r

                # The response is a bullet point list of moves. Use regex
                matches = re.findall(r"- ([\w ]+)", llm_response)
                moves = ["".join(match) for match in matches]
                invalid_moves = []
                valid_moves = []
                for move in moves:
                    cleaned_move_name = move.strip().lower()
                    if cleaned_move_name in META_INSTRUCTIONS_WITH_LOWER.keys():
                        if self.player_nb == 1:
                            print(
                                f"[red] Player {self.player_nb} move: {cleaned_move_name}"
                            )
                        elif self.player_nb == 2:
                            print(
                                f"[green] Player {self.player_nb} move: {cleaned_move_name}"
                            )
                        valid_moves.append(cleaned_move_name)
                    else:
                        logger.debug(f"Invalid completion: {move}")
                        logger.debug(f"Cleaned move name: {cleaned_move_name}")
                        invalid_moves.append(move)

                if len(invalid_moves) > 1:
                    self.grounding_errors+=1
                    logger.warning(f"Many invalid moves: {invalid_moves}")

            logger.debug(f"Next moves: {valid_moves}")
            return valid_moves

    def call_llm(
        self,
        temperature: float = 0.7,
        max_tokens: int = 50,
        top_p: float = 1.0,
    ) -> str:
        """
        Make an API call to the language model.

        Edit this method to change the behavior of the robot!
        """

        # Generate the prompts
        move_list = "- " + "\n - ".join([move for move in META_INSTRUCTIONS])

        # print("===========================")
        # print(move_list)
        # print("===========================")
        system_prompt = self.prompt_templete.sys_prompt.format(character=self.character,context_prompt=self.context_prompt(),move_list=move_list)
        start_time = time.time()

        messages = [
            {"role":"system", "content":system_prompt},
        {"role":"user", "content":"Your next moves are:"}
        ]
        # weave.init(self.weave_prj_name)
        # resp=self.model.query_single_turn(messages)
        # resp = resp.choices[0].message.content

        # resp=self.model.chat_llm(messages)
        resp=self.llm_query(messages)

        self.logger.info(f"=============model:{self.model.model_name}===============")
        self.logger.info(f"====model_input:{messages}")
        self.logger.info(f"====model_output:{resp}")

        ## trajectory
        state_info = set_state_info(from_="Streetfight3", role=self.role, step=self.cur_time_step, content=system_prompt+"Your next moves are:",
                                    system_content=system_prompt, user_content="Your next moves are:")
        self.trajectory.append(state_info)
        action = set_action_info(from_=self.model.model_name, role=self.role, step=self.cur_time_step, content=resp,
                                 other_content={})
        self.trajectory.append(action)

        print("self.player_nb:",self.player_nb)
        print("self.model:",self.model.model_name)
        print("ret:",resp)
        # print("type:",type(messages))
        # print("=====:",messages)
        print("~~~~")
        logger.debug(f"LLM call to {self.model}: {system_prompt}")
        logger.debug(f"LLM call to {self.model}: {time.time() - start_time}s")

        return resp

    def llm_query(self,messages):
        if self.model.model_name == 'o1-mini':
            messages[0]['role'] = 'user'
            resp,_ = self.model.query_single_turn_o1(messages)

        else:
            resp = self.model.chat_llm(messages)

        return  resp