# Copyright (C) 2023  The CivRealm project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
import json
import os
import time
import threading

from .baselang_agent import BaseLangAgent
from .workers import MastabaWorker
from agent_manager.agents.civ_agent.config import INDIVIDUAL_PROMPT_DEFAULT, PROMPT_SOLUTIONS
from agent_manager.agents.trajectory import Trajectory,set_action_info,set_state_info,set_reward

class MastabaAgent(BaseLangAgent):
    def __init__(
            self, config,args,
            use_entity_individual_prompt: bool = INDIVIDUAL_PROMPT_DEFAULT,
            **kwargs):
        self.prompt_constructor = config.prompt
        self.model = config.llm_model
        self.args=args
        self.logger = self.args.logger
        print("====MastabaAgent====")
        super().__init__(**kwargs)
        self.use_entity_individual_prompt = use_entity_individual_prompt
        self.general_advise = ""
        self.trajectory: Trajectory = []
        self.logger.info("=" * 5 + f"MastabaAgent Init Successfully!: " + "=" * 5)


    def initialize_workers(self):
        self.strategy_maker = MastabaWorker(self.args,llm=self.model,role="advisor")
        self.workers = {}

    def add_entity(self, entity_type, entity_id):

        if self.use_entity_individual_prompt:
            name = self.info['llm_info'][entity_type][entity_id]['name']
            name = name.split(" ")[0]

            prompt_prefix = PROMPT_SOLUTIONS[name]
        else:
            prompt_prefix = PROMPT_SOLUTIONS['vanilla']
        self.workers[(entity_type,
                      entity_id)] = MastabaWorker(self.args,llm=self.model,ctrl_type=entity_type,
                                                  actor_id=entity_id,
                                                  prompt_prefix=prompt_prefix)

    def get_advisor_input_prompt(self, obs, info):
        """
        Generate input prompt for advisor.
        """
        munit_num_self = 0  # millitary units
        wunit_num_self = 0  # working units
        unit_num_enemy = 0
        city_num_self = 0
        city_size_self = 0
        city_num_other = 0
        city_num_enemy = 0
        units = {}
        # add ['self_id'] to info['llm_info']

        for key, val in obs['unit'].items():
            if val['owner'] == info['my_player_id']:
                unit_name = val['type_rule_name']
                units[unit_name] = units.get(unit_name, 0) + 1
                if val['type_attack_strength'] == 0:
                    wunit_num_self += 1
                else:
                    munit_num_self += 1

            if obs['dipl'][val['owner']]['diplomatic_state'] == 1:
                unit_num_enemy += 1

        for key, val in obs['city'].items():
            if val['granary_size'] >= 0:
                city_num_self += 1
                city_size_self += val['size']
                continue
            if obs['map']['status'][val['x'], val['y']] <= 1:
                # if a city is not in current view (but in war fog)
                # it is not counted.
                continue
            if obs['dipl'][val['owner']]['diplomatic_state'] == 1:
                city_num_enemy += 1
                continue
            city_num_other += 1

        # handwritten conditions, change it later.
        if (unit_num_enemy > munit_num_self / 5
                and city_num_enemy < unit_num_enemy):
            war_state = "We are under attack."
        elif city_num_enemy >= 3 and unit_num_enemy < munit_num_self:
            war_state = "We are attacking other players."
        elif unit_num_enemy == 0 and city_num_enemy < 4:
            war_state = "We are in peace."
        else:
            war_state = "We are roughly safe."

        unit_spec_prompt = (
            f"We have {wunit_num_self+munit_num_self} units: " +
            ", ".join(map(lambda x: f"{x[1]} {x[0]}", units.items())))
        return_prompt = " ".join([
            unit_spec_prompt, f"and we can see {unit_num_enemy} enemy units.",
            f"We have {city_num_self} cities of total size {city_size_self}.",
            f"We can see {city_num_enemy} enemy cities, ",
            f"and {city_num_other} other cities.", war_state
        ])
        return return_prompt

    def get_obs_input_prompt(self, ctrl_type, actor_name, actor_dict,
                             available_actions):
        zoom_in_obs = actor_dict['observations']['minimap']
        zoom_out_obs = actor_dict['observations']['upper_map']
        system_message = self.info['llm_info'].get("message", "")
        system_message = ("Game scenario message is: "
                          if system_message else "") + system_message
        prompt = ""

        if ctrl_type == "city":
            producing = actor_dict['observations'].get('producing', "NOTHING")
            available_actions += ['produce ' + producing]
            # print("===+++=== PRODUCING", producing)
            prompt = self.strategy_maker.prompt_handler.city_obs_action(
                actor_name=actor_name,
                ctrl_type=ctrl_type,
                zoom_out_obs=zoom_out_obs,
                zoom_in_obs=zoom_in_obs,
                producint=producing,
                available_actions=available_actions,
                general_advise=self.general_advise)
        elif ctrl_type == "unit":
            prompt = self.strategy_maker.prompt_handler.unit_obs_action(
                actor_name=actor_name,
                ctrl_type=ctrl_type,
                zoom_out_obs=zoom_out_obs,
                zoom_in_obs=zoom_in_obs,
                available_actions=available_actions,
                general_advise=self.general_advise)
        return prompt

    def generate_general_advise(self):
        """
        Generate general advise for all other workers.
        """

        obs_input_prompt = self.get_advisor_input_prompt(
            self.observations, self.info)

        # obs_input_prompt = self.strategy_maker.prompt_handler.generate(
        #     "advisor_advise")
        # print("OBS_INPUT_PROMPT", obs_input_prompt)
        exec_action_name = self.strategy_maker.choose_action(
            obs_input_prompt, ["suggestion"])
        # print("GENERAL_ADVISE", exec_action_name)
        ## trajectory
        state_info = set_state_info(from_="Civilization", role="player", step=self.info['turn'],
                                    content=self.strategy_maker.dialogue[2]['content'],
                                    system_content=self.strategy_maker.dialogue[0]['content']+self.strategy_maker.dialogue[1]['content'], user_content=self.strategy_maker.dialogue[2]['content'])
        self.trajectory.append(state_info)
        action = set_action_info(from_=self.strategy_maker.llm.model_name, role="player", step=self.info['turn'],
                                 content=exec_action_name, other_content=self.strategy_maker.dialogue[-1]['content'])
        self.trajectory.append(action)
        return exec_action_name

    def make_decisions(self):
        if self.is_new_turn:
            self.general_advise = self.generate_general_advise()

        threads = []
        for ctrl_type in self.info['llm_info'].keys():
            for actor_id, actor_dict in self.info['llm_info'][ctrl_type].items(
            ):
                if (self.last_taken_actions.get(
                    (ctrl_type, actor_id), ["", -2])[1] == self.turn
                        and ctrl_type != "unit"):
                    print(
                        f"{ctrl_type} {actor_id} tries a second move but rejected."
                    )
                    continue
                thread = threading.Thread(target=self.make_single_decision,
                                          args=(ctrl_type, actor_id,
                                                actor_dict))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        # super().make_decisions()

    def handle_conflict_actions(self, action):
        """
        Handle conflict actions by adding it into list
        `self.conflict_action_list`.
        """
        self.conflict_action_list += [action]

    def regenerate_conflict_actions(self, observations, info):
        """Follow a similar logic of `make_decisions`."""

        print("Regenerating_conflict_actions")
        print(self.chosen_actions.qsize(), self.current_deconflict_depth)
        # super().regenerate_conflict_actions(observations, info)
        self.handle_new_turn(observations, info)

    def save_trajectory(self,role, save_path,name):
        item_id=name.split("/")[-1].strip()+"_"+role
        output_path = os.path.join(save_path,item_id+".json")
        temp_traj=[]
        for traj in self.trajectory:
            if traj["role"]==role:
                temp_traj.append(traj)
        react_data={"item_id":item_id,"conversation":temp_traj}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(react_data, f, ensure_ascii=False, indent=2)
