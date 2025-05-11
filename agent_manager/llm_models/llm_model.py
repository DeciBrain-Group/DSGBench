import weave
# from langchain.chat_models import ChatOpenAI, ChatAnyscale
from langchain_community.chat_models import ChatOpenAI, ChatAnyscale
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import openai
import time
import random
import dashscope
from http import HTTPStatus
import ollama
from openai import OpenAI



class LLMModel(object):
    def __init__(self, args):
        self.model_name = args.model_name
        self.api_key=args.api_key
        self.api_url=args.api_url
        self.max_tokens = args.max_tokens
        self.timeout = args.timeout
        self.temperature = args.temperature

    @weave.op()
    def query_single_turn(self,input,n=10):
        client=OpenAI(api_key=self.api_key,base_url=self.api_url)
        run_time=0
        while run_time<n:
            try:
                output = client.chat.completions.create(
                    model=self.model_name,
                    messages=input,
                    temperature=self.temperature
                )
                if output.choices is None:
                    print(self.model_name," return output is None:",output)
                    run_time+=1
                    continue
                else:
                    return output
            except Exception as e:
                print(f"==============Error when calling the {self.model_name} API: {e}")
                time.sleep(2)
                run_time += 1

    def query_single_turn_gen(self, messages):
        if self.model_name == 'o1-mini':
            messages[0]['role'] = 'user'
            resp,output = self.query_single_turn_o1(messages)
            usage=output.usage

        else:
            response = self.query_single_turn(messages)
            resp = response.choices[0].message.content
            usage =response.usage
        return resp,usage

    def query_single_turn_o1(self,input,n=10):
        client=OpenAI(api_key=self.api_key,base_url=self.api_url)
        run_time=0
        while run_time<n:
            try:
                output = client.chat.completions.create(
                    model=self.model_name,
                    messages=input,
                    max_completion_tokens=4096,
                    timeout=600
                )
                if output.choices is None:
                    print(self.model_name," return output is None:",output)
                    run_time+=1
                    continue
                else:
                    response=output.choices[0].message.content
                    return response,output
            except Exception as e:
                # Output error message
                print(f"Error when calling the OpenAI API: {e}")
                time.sleep(2)
                run_time+=1

    @weave.op()
    def query(self, user_input, n=1,stop=None):
        """
        query method, used for conversation
        :param user_input: User input
        :return: Bot response
        """
        if self.model_name.__contains__("gpt"):
            chat = ChatOpenAI(model_name=self.model_name,
                              openai_api_key=self.api_key,
                              temperature=self.temperature,
                              max_tokens=self.max_tokens,
                              n=n,
                              request_timeout=self.timeout,
                              openai_api_base=self.api_url
                              )
        elif 'Open-Orca/Mistral-7B-OpenOrca' == self.model_name:
            chat = ChatAnyscale(temperature=self.temperature,
                                anyscale_api_key=self.api_key,
                                max_tokens=self.max_tokens,
                                n=n,
                                model_name=self.model_name,
                                request_timeout=self.timeout)
        else:
            # deepinfra
            chat = ChatOpenAI(model_name=self.model_name,
                              openai_api_key=self.api_key,
                              temperature=self.temperature,
                              max_tokens=self.max_tokens,
                              n=n,
                              request_timeout=self.timeout,
                              openai_api_base=self.api_url)

        longchain_msgs = []
        for msg in user_input:
            if msg['role'] == 'system':
                longchain_msgs.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'user':
                longchain_msgs.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                longchain_msgs.append(AIMessage(content=msg['content']))
            else:
                raise NotImplementedError

        # Try to send request and get response
        for retries in range(n):
            try:
                generations = chat.generate([longchain_msgs], stop=[stop] if stop is not None else None)
                responses = [chat_gen.message.content for chat_gen in generations.generations[0]]

                return responses[0]
            except Exception as e:
                # Output error message
                print(f"Error when calling the OpenAI API: {e}")

                # If maximum retry attempts reached, return a specific response
                if retries >= n - 1:
                    print("Maximum number of retries reached. The API is not responding.")
                    return "I'm sorry, but I am unable to provide a response at this time due to technical difficulties."

                # Wait for a period before retrying, using exponential backoff strategy
                sleep_time = (2 ** retries) + random.random()
                print(f"Waiting for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)

    @weave.op()
    def chat_llm(self, messages, n=1, stop=None):
        if self.model_name.__contains__("gpt"):
            chat = ChatOpenAI(model_name=self.model_name,
                              openai_api_key=self.api_key,
                              temperature=self.temperature,
                              max_tokens=self.max_tokens,
                              n=n,
                              request_timeout=self.timeout,
                              openai_api_base=self.api_url
                              )
        elif 'Open-Orca/Mistral-7B-OpenOrca' == self.model_name:
            chat = ChatAnyscale(temperature=self.temperature,
                                anyscale_api_key=self.api_key,
                                max_tokens=self.max_tokens,
                                n=n,
                                model_name=self.model_name,
                                request_timeout=self.timeout)
        else:
            # deepinfra
            chat = ChatOpenAI(model_name=self.model_name,
                              openai_api_key=self.api_key,
                              temperature=self.temperature,
                              max_tokens=self.max_tokens,
                              n=n,
                              request_timeout=self.timeout,
                              openai_api_base=self.api_url)

        longchain_msgs = []
        for msg in messages:
            if msg['role'] == 'system':
                longchain_msgs.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'user':
                longchain_msgs.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                longchain_msgs.append(AIMessage(content=msg['content']))
            else:
                raise NotImplementedError

        if n > 1:
            response_list = []
            total_completion_tokens = 0
            total_prompt_tokens = 0
            for n in range(n):
                generations = chat.generate([longchain_msgs], stop=[stop] if stop is not None else None)
                responses = [
                    chat_gen.message.content for chat_gen in generations.generations[0]]
                response_list.append(responses[0])
                completion_tokens = generations.llm_output['token_usage']['completion_tokens']
                prompt_tokens = generations.llm_output['token_usage']['prompt_tokens']
                total_completion_tokens += completion_tokens
                total_prompt_tokens += prompt_tokens
            responses = response_list
            completion_tokens = total_completion_tokens
            prompt_tokens = total_prompt_tokens
        else:
            generations = chat.generate([longchain_msgs], stop=[
                stop] if stop is not None else None)
            responses = [
                chat_gen.message.content for chat_gen in generations.generations[0]]
            # completion_tokens = generations.llm_output['token_usage']['completion_tokens']
            # prompt_tokens = generations.llm_output['token_usage']['prompt_tokens']

        return responses[0]


class AliLLM(object):
    def __init__(self, args):
        self.model_name = args.model_name
        self.api_key = args.api_key
        dashscope.api_key = self.api_key
        self.api_url = args.api_url
        self.max_tokens = args.max_tokens
        self.timeout = args.timeout
        self.temperature = args.temperature

    def query(self, user_input, n=1, stop=None):
        """
        query method, used for conversation
        :param user_input: User input
        :return: Bot response
        """
        # Try to send request and get response
        for retries in range(n):
            try:
                response = dashscope.Generation.call(
                    model=self.model_name,
                    messages=user_input,
                    seed=random.randint(1, 10000),
                    max_tokens=self.max_tokens,
                    result_format='message',  # set the result to be "message" format.
                )
                if response.status_code == HTTPStatus.OK:
                    response = response.output.choices[0].message.content
                    return response
                else:
                    raise Exception("response.status_code is not normal:{}".format(response.status_code))
            except Exception as e:
                # Output error message
                print(f"Error when calling the OpenAI API: {e}")

                # If maximum retry attempts reached, return a specific response
                if retries >= n - 1:
                    print("Maximum number of retries reached. The API is not responding.")
                    return "I'm sorry, but I am unable to provide a response at this time due to technical difficulties."

                # Wait for a period before retrying, using exponential backoff strategy
                sleep_time = (2 ** retries) + random.random()
                print(f"Waiting for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
        return ""

class OLLAMALLM(object):
    def __init__(self, args):
        self.model_name = args.model_name
        self.api_key = args.api_key
        self.api_url = args.api_url
        self.max_tokens = args.max_tokens
        self.timeout = args.timeout
        self.temperature = args.temperature
        self.client = ollama.Client(host=self.api_url)

    def query(self, user_input, n=1, stop=None):
        """
        query method, used for conversation
        :param user_input: User input
        :return: Bot response
        """
        # Try to send request and get response
        for retries in range(n):
            try:
                response = (self.client.chat(
                    model=self.model_name,
                    messages=user_input
                ))

                response = response['message']['content']
                return response
            except Exception as e:
                # Output error message
                print(f"Error when calling the OpenAI API: {e}")

                # If maximum retry attempts reached, return a specific response
                if retries >= n - 1:
                    print("Maximum number of retries reached. The API is not responding.")
                    return "I'm sorry, but I am unable to provide a response at this time due to technical difficulties."

                # Wait for a period before retrying, using exponential backoff strategy
                sleep_time = (2 ** retries) + random.random()
                print(f"Waiting for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
        return ""

if __name__ == '__main__':

    from box import Box
    import yaml

    user_input = [
        {"role": "system", "content": "System initialized."},
        {"role": "user", "content": "What is the current status?"},
        {"role": "assistant", "content": "The system is operational."}
    ]
    llm_path= "../../configs/llm_configs/gpt-35-turbo-0125.yaml"
    llm_config = Box.from_yaml(filename=llm_path, Loader=yaml.FullLoader)
    llm_model=LLMModel(llm_config)
    print(llm_model.query(user_input, n=1,stop=None))
    # print(llm_model.query_single_turn(user_input))
    print("====")