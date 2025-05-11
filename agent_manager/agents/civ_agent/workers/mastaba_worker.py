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
import pinecone
import weave

from .gpt_worker import AzureGPTWorker
from langchain_community.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Pinecone
class MastabaWorker(AzureGPTWorker):
    def __init__(self,args,llm,role="controller", **kwargs):
        self.args=args
        self.role = role
        self.llm=llm
        super().__init__(args,role,**kwargs)
        self.logger = self.args.logger
        actor_str="" if not kwargs else str(kwargs.get("actor_id",""))
        actor_str=",entity_id: "+actor_str if len(actor_str)>0 else ""
        self.logger.info("=" * 5 + f"role: {self.role} {actor_str}, MastabaWorker Init Successfully!: " + "=" * 5)


    def init_llm(self):
        llm = ChatOpenAI(model_name=self.llm.model_name,
                          openai_api_key=self.llm.api_key,
                          temperature=self.llm.temperature,
                          max_tokens=self.llm.max_tokens,
                          n=1,
                          request_timeout=self.llm.timeout,
                          openai_api_base=self.llm.api_url
                          )
        self.chain = load_qa_chain(llm,chain_type="stuff")
        self.memory = ConversationSummaryBufferMemory(llm=llm,
                                                      max_token_limit=500)

    def init_index(self):
        pinecone.init(api_key=self.args.agent[0].pinecone_key)
        # self.index = Pinecone.from_existing_index(
        #     index_name='civrealm-mastaba',
        #     embedding=OpenAIEmbeddings(openai_api_key='sk-sufonZ4mNq9h9SDq7e311aA6C31c40A1BaB5369481D96cFf',openai_proxy='https://api.openai.com',model="text-embedding-ada-002"))
        self.index = Pinecone(index=pinecone.Index('civrealm-mastaba'),
                              embedding=OpenAIEmbeddings(
                                  openai_api_key=self.args.agent[0].agent_embedding_api_key,
                                  openai_proxy=self.args.agent[0].agent_embedding_api_url, model=self.args.agent[0].agent_embedding_model_name),
                              text_key="text")
    def _load_instruction_prompt(self):
        if self.role == "controller":
            instruction_prompt = self.prompt_handler.hierarchical_instruction_prompt(
            )

        else:
            instruction_prompt = self.prompt_handler.advisor_instruction_nomap(
            )
            pass
        self.add_user_message_to_dialogue(instruction_prompt)

    def _load_task_prompt(self):
        if self.role == "controller":
            task_prompt = self.prompt_handler.hierarchical_task_prompt()
        else:
            task_prompt = self.prompt_handler.advisor_task()
        self.add_user_message_to_dialogue(task_prompt)
