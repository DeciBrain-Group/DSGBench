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

import os
import random
import json
import pinecone

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

from civrealm.freeciv.utils.freeciv_logging import fc_logger
from agent_manager.agents.civ_agent.prompt_handlers.base_prompt_handler import BasePromptHandler

# from .base_worker import BaseWorker
from agent_manager.agents.civ_agent.workers.base_worker import BaseWorker

class AzureGPTWorker(BaseWorker):
    """
    This agent uses GPT-3 to generate actions.
    """
    def __init__(self,args,role="controller",
                 model: str = 'gpt-35-turbo-16k',
                 prompt_prefix: str = "civ_prompts",
                 **kwargs):
        # assert os.environ['OPENAI_API_TYPE'] == 'azure'
        self.args = args
        self.logger = self.args.logger
        self.role = role
        self.entity_id = "" if not kwargs else str(kwargs.get("actor_id", ""))
        self.prompt_prefix = prompt_prefix
        super().__init__(model, **kwargs)

    def init_prompts(self):
        self.prompt_handler = BasePromptHandler(
            prompt_prefix=self.prompt_prefix)
        self._load_instruction_prompt()
        self._load_task_prompt()

    def init_index(self):
        pinecone.init(api_key='4dc46cc0-9dda-4049-aeab-cfc2962ae192')
                      # environment=os.environ["MY_PINECONE_ENV"]
        # print(os.environ)
        # self.index = Pinecone.from_existing_index(
        #     index_name='civrealm-mastaba',
        #     embedding=OpenAIEmbeddings(openai_api_key='sk-sufonZ4mNq9h9SDq7e311aA6C31c40A1BaB5369481D96cFf',openai_proxy='https://api.openai.com',model="text-embedding-ada-002"))
        self.index=Pinecone(index=pinecone.Index('civrealm-mastaba'),
                            embedding=OpenAIEmbeddings(openai_api_key='sk-sufonZ4mNq9h9SDq7e311aA6C31c40A1BaB5369481D96cFf',openai_proxy='https://api.openai.com',model="text-embedding-ada-002"),
                            text_key="text")


    def _load_instruction_prompt(self):
        instruction_prompt = self.prompt_handler.instruction_prompt()
        self.add_user_message_to_dialogue(instruction_prompt)

    def _load_task_prompt(self):
        task_prompt = self.prompt_handler.task_prompt()
        self.add_user_message_to_dialogue(task_prompt)

    def register_all_commands(self):
        self.register_command('manualAndHistorySearch',
                              self.handle_command_manual_and_history_search)
        self.register_command('finalDecision',
                              self.handle_command_final_decision)
        self.register_command('suggestion', self.handle_command_suggestion)

    def handle_command_manual_and_history_search(self, command_input,
                                                 obs_input_prompt,
                                                 current_avail_actions):
        if self.taken_actions_list_needs_update('look_up', 0, 0):
            answer = self.prompt_handler.finish_look_for()
            self.add_user_message_to_dialogue(answer)
            return None, ''

        query = command_input['look_up']
        answer = self.get_answer_from_index(query) + '.\n'
        if random.random() > 0.5:
            answer += self.prompt_handler.finish_look_for()

        self.add_user_message_to_dialogue(answer)
        self.memory.save_context({'assistant': query}, {'user': answer})
        self.taken_actions_list.append('look_up')
        return None, ''

    def handle_command_ask_current_game_information(self, command_input,
                                                    obs_input_prompt,
                                                    current_avail_actions):
        self.taken_actions_list.append('askCurrentGameInformation')
        return None, ''

    def handle_command_suggestion(self, command_input, obs_input_prompt,
                                  current_avail_actions):
        exec_action = command_input["suggestion"]
        return exec_action, ''

    def handle_command_final_decision(self, command_input, obs_input_prompt,
                                      current_avail_actions):
        exec_action = command_input['action']
        lower_avail_actions = [x.lower() for x in current_avail_actions]
        if exec_action.lower() not in lower_avail_actions:
            print(f'{self.name}\'s chosen action "{exec_action}" not in the ' +
                  f'available action list, available actions are ' +
                  f'{current_avail_actions}, retrying...')
            fc_logger.error(
                f'{self.name}\'s chosen action "{exec_action}"',
                'not in the available action list, available',
                f'actions are {current_avail_actions}, retrying...')
            return None, self.prompt_handler.insist_avail_action()

        self.taken_actions_list.append(command_input['action'])

        for move_name in current_avail_actions:
            if move_name[:4] != "move":
                continue
            if self.taken_actions_list_needs_update(move_name, 15, 4):
                return None, self.prompt_handler.insist_various_actions(
                    action=move_name)

        return exec_action, ''

    def query_llm(self, stop=None, temperature=0.7, top_p=0.95):
        fc_logger.debug(f'Querying with dialogue: {self.dialogue}')
        # print(self.dialogue)
        if self.llm.model_name == 'o1-mini':
            chunks_str=self.dialogue
            chunks_str[0]['role'] = 'user'
            output, _ = self.llm.query_single_turn_o1(chunks_str)
        else:
            output = self.llm.query(self.dialogue)
        self.logger.info(f"=============model:{self.llm.model_name}===============")
        self.logger.info(f"=============role:{self.role}--{self.entity_id}=============")
        self.logger.info(f"====model_input:{self.dialogue}")
        self.logger.info(f"====model_output:{output}")
        return output

    def generate_command(self, prompt: str):
        self.add_user_message_to_dialogue(prompt +
                                          self.prompt_handler.insist_json())
        self.restrict_dialogue()
        response = self.query_llm()
        # self.memory.save_context({'user': prompt},
        #                          {'assistant': str(response)})
        return response

    def parse_response(self, response):
        # content = response.choices[0].message.content # for openai==1.45.0
        content = response
        start_index = content.find('{')
        end_index = content.rfind('}') + 1
        rlack = content.count("{") - content.count("}")
        if rlack > 0:
            content = content[start_index:end_index] + "}" * rlack
        else:
            content = content[start_index:end_index]
        return json.loads(content)

    def process_command(self, response, obs_input_prompt,
                        current_avail_actions):
        # First try to parse the reponse by the given json format
        fc_logger.debug(f'Processing response: {response}')
        try:
            command_json = self.parse_response(response)
            command_input = command_json['command']['input']
            command_name = command_json['command']['name']
        except Exception as e:
            fc_logger.error(
                f'\nRESPONSE:{response}\nCommond json parsing error: {e}')
            print('Not in given json format, retrying...')
            return None, self.prompt_handler.insist_json()

        # Then check if the command is valid
        if command_name not in self.command_handlers:
            fc_logger.error(f'Unknown command: {command_name}')
            available_commands = ', '.join(self.command_handlers.keys())
            prompt_addition = self.prompt_handler.insist_available_commands(
                available_commands)
            return None, prompt_addition

        return self.command_handlers[command_name](command_input,
                                                   obs_input_prompt,
                                                   current_avail_actions)

if __name__ == '__main__':
    gpt = AzureGPTWorker()
    print(gpt.query_llm())