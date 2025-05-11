import os
from datetime import datetime
import multiprocessing

import warnings
warnings.filterwarnings("ignore")
import wandb
import weave
import yaml
from box import Box
import argparse
from agent_eval.agent_eval import AgentEval
from utils.agent_logger import AgentLogger, TEST_LOGGER
from tasks_config import TASKS,WANDB_API_KEY,WANDB_TIMEOUT

MAX_PROCESS_NUM=1
MAX_PROCESS_MATCH_NUM=1
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ['WANDB_TIMEOUT'] = WANDB_TIMEOUT



def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("--tasks", required=False, type=str, nargs='+', default=["all"],help="specify the tasks")
    parser.add_argument("--cfg_path", required=False, default='configs/eval_configs/',
                        help="specify the place to store the resuls")
    parser.add_argument("--runs_log_path", required=False, default='./output/runs_log.txt',help="specify the place to store the resuls")
    args = parser.parse_args()

    return args


class TaskMultiEval(object):
    def __init__(self, comm_args, task):
        self.logger = None
        self.agent_eval = None
        self.args = None
        self.success_rate = 0.0
        self.grounding_acc = 0.0
        self.opp_success_rate = 0.0
        self.opp_grounding_acc = 0.0
        self.comm_args = comm_args
        self.task = task
        self.config_path = os.path.join(self.comm_args.cfg_path, TASKS[self.task])
        self.args = Box.from_yaml(filename=self.config_path, Loader=yaml.FullLoader)
        if not os.path.exists(self.args.eval.output_path):
            os.makedirs(self.args.eval.output_path,)
        assert self.args.eval.num_matches >= 1


    def judge_task_exec_status(self):
        api = wandb.Api()
        # get task runs
        runs = api.runs(self.args.eval.weave_prj_name, filters={'state': 'finished'})
        try:
            valid_runs_num=len(runs)
        except:
            valid_runs_num=0
        next_runs_num=0
        if valid_runs_num>=self.args.eval.num_matches:
            next_runs_num=self.args.eval.num_matches
        else:
            next_runs_num=valid_runs_num
        return next_runs_num

    def task_reset(self, match_idx):
        self.logger = AgentLogger(name=TEST_LOGGER, filepath=os.path.join(self.args.eval.output_path, self.task +"_"+str(match_idx)+ ".log"))
        self.args.eval_config = self.config_path
        self.args.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.args.match_idx = match_idx
        self.args.logger = self.logger
        self.agent_eval = AgentEval(self.args)

        self.logger.info("=" * 20)
        self.logger.info("=" * 20)
        self.logger.info("=" * 5 + f" log beginning: task: {self.task} " + "=" * 5)
        self.logger.info("=" * 5 + f"reset match : {match_idx} " + "=" * 5)

    def task_run(self):
        next_runs_num=self.judge_task_exec_status()
        print(f"=========next_runs_num============={next_runs_num}")
        # next_runs_num=0
        ret_all = []
        match_idx=next_runs_num
        error_times=0
        while match_idx<=self.args.eval.num_matches:
            # try:
            print("beginning match:",match_idx)
            self.task_reset(match_idx)
            self.agent_eval.play()
            eval_ret = self.agent_eval.get_eval_result()
            print("~~~~~~~~~~~~~~~~~~~~~~~")
            print(eval_ret)
            print("~~~~~~~~~~~~~~~~~~~~~~~")
            ret_all.append(eval_ret)
            if len(ret_all) >= 1:
                self.calc_eval_ret(ret_all)
            self.agent_eval.close()
            match_idx+=1
            error_times=0
            # except Exception as e:
            #     self.logger.info(f"error:{e}")
            #     print(e)
            #     error_times+=1
            #     print("keep error times:",error_times)
            #     if error_times>5:
            #         match_idx+=1
            #         error_times=0
            #         print("keep error times reach 5 times, go to the next match: ",match_idx)
            #     continue



    def calc_eval_ret(self, ret_all,store=True):
        if self.args.game.game_name.__contains__("Starcraft"):
            player_win_all = 0
            grounding_acc_all = 0
            match_PBR_all = 0
            match_RUR_all = 0
            match_APU_all = 0
            match_TR_all = 0
            match_num = len(ret_all)
            for ret in ret_all:
                if ret['result'] == 1:
                    player_win_all += 1
                grounding_acc_all += ret['grounding_acc']
                match_PBR_all += ret['PBR']
                match_RUR_all += ret['RUR']
                match_APU_all += ret['APU']
                match_TR_all += ret['TR']

            player_win_rate = player_win_all / match_num
            grounding_acc_avg = grounding_acc_all / match_num
            match_PBR_avg = match_PBR_all / match_num
            match_RUR_avg = match_RUR_all / match_num
            match_APU_avg = match_APU_all / match_num
            match_TR_avg = match_TR_all / match_num
            task_result = {"player_win_rate": player_win_rate,
                           "grounding_acc_avg": grounding_acc_avg,
                           "match_PBR_avg": match_PBR_avg,
                           "match_RUR_avg": match_RUR_avg,
                           "match_APU_avg": match_APU_avg,
                           "match_TR_avg": match_TR_avg
                           }
            wandb.log(task_result)
        if self.args.game.game_name.__contains__("Stratego"):
            player_win_all = 0
            opp_player_win_all = 0
            match_turns_all=0
            grounding_acc_all=0
            opp_grounding_acc_all=0
            match_live_pieces_rate_all=0
            match_live_pieces_score_all=0
            match_opp_live_pieces_rate_all=0
            match_opp_live_pieces_score_all=0
            match_num=len(ret_all)
            for ret in ret_all:
                if ret['player'] == 1:
                    player_win_all += 1
                if ret['opp_player'] == 1:
                    opp_player_win_all += 1
                match_turns_all+=ret['match_turns']
                grounding_acc_all+=ret['grounding_acc']
                opp_grounding_acc_all+=ret['opp_grounding_acc']
                match_live_pieces_rate_all+=ret['match_live_pieces_rate']
                match_live_pieces_score_all+=ret['match_live_pieces_score']
                match_opp_live_pieces_rate_all+=ret['match_opp_live_pieces_rate']
                match_opp_live_pieces_score_all+=ret['match_opp_live_pieces_score']

            player_win_rate=player_win_all/match_num
            opp_player_win_rate=opp_player_win_all/match_num
            match_turns_avg=match_turns_all/match_num
            grounding_acc_avg=grounding_acc_all/match_num
            opp_grounding_acc_avg=opp_grounding_acc_all/match_num
            match_live_pieces_rate_avg=match_live_pieces_rate_all/match_num
            match_live_pieces_score_avg=match_live_pieces_score_all/match_num
            match_opp_live_pieces_rate_avg=match_opp_live_pieces_rate_all/match_num
            match_opp_live_pieces_score_avg=match_opp_live_pieces_score_all/match_num
            task_result={"player_win_rate": player_win_rate,
                       "opp_player_win_rate": opp_player_win_rate,
                       "match_turns_avg": match_turns_avg,
                       "grounding_acc_avg": grounding_acc_avg,
                       "opp_grounding_acc_avg": opp_grounding_acc_avg,
                       "match_live_pieces_rate_avg": match_live_pieces_rate_avg,
                       "match_live_pieces_score_avg": match_live_pieces_score_avg,
                       "match_opp_live_pieces_rate_avg": match_opp_live_pieces_rate_avg,
                       "match_opp_live_pieces_score_avg": match_opp_live_pieces_score_avg,
                       }
            wandb.log(task_result)
        if self.args.game.game_name.__contains__("StreetFight"):
            player_win_all = 0
            opp_player_win_all = 0
            match_turns_all=0
            grounding_acc_all=0
            opp_grounding_acc_all=0
            match_time_use_all=0
            match_num=len(ret_all)
            for ret in ret_all:
                if ret['player'] == 1:
                    player_win_all += 1
                if ret['opp_player'] == 1:
                    opp_player_win_all += 1
                match_turns_all+=ret['match_turns']
                grounding_acc_all+=ret['grounding_acc']
                opp_grounding_acc_all+=ret['opp_grounding_acc']
                match_time_use_all+=ret['match_time_use']

            player_win_rate=player_win_all/match_num
            opp_player_win_rate=opp_player_win_all/match_num
            match_turns_avg=match_turns_all/match_num
            grounding_acc_avg=grounding_acc_all/match_num
            opp_grounding_acc_avg=opp_grounding_acc_all/match_num
            match_time_use_avg=match_time_use_all/match_num
            task_result={"player_win_rate": player_win_rate,
                       "opp_player_win_rate": opp_player_win_rate,
                       "match_turns_avg": match_turns_avg,
                       "grounding_acc_avg": grounding_acc_avg,
                       "opp_grounding_acc_avg": opp_grounding_acc_avg,
                       "match_time_use": match_time_use_avg,
                       }
            wandb.log(task_result)
        if self.args.game.game_name.__contains__("WereWolf"):
            player_win_all = 0
            match_turns_all=0
            match_num=len(ret_all)
            for ret in ret_all:
                if ret['player'] == 1:
                    player_win_all += 1
                match_turns_all+=ret['match_turns']

            player_win_rate=player_win_all/match_num
            match_turns_avg=match_turns_all/match_num
            task_result={"player_win_rate": player_win_rate,
                       "match_turns_avg": match_turns_avg
                       }
            wandb.log(task_result)


def record_runs(task,log_path):
    api = wandb.Api()
    # get task runs
    runs = api.runs(task, filters={'state': 'finished'})
    try:
        valid_runs_num = len(runs)
    except:
        valid_runs_num = 0
    task_run_str=f"{task}  {valid_runs_num} \n"
    with open(log_path,'a+') as f:
        f.write(task_run_str)
def main():
    comm_args = parse_args()
    task_names = comm_args.tasks if comm_args.tasks != ["all"] else TASKS.keys()
    for task in task_names:
        print("task========",task)
        task_eval=TaskMultiEval(comm_args,task)
        task_eval.task_run()

def one_task_eval(task,runs_log_path):
    print("Begging task : {}".format(task))
    comm_args = parse_args()
    task_eval = TaskMultiEval(comm_args, task)
    task_eval.task_run()
    print("Done task : {} ".format(task))
    record_runs(task, runs_log_path)


def multi_process_task_run():
    comm_args = parse_args()
    cur_time=datetime.now().strftime('%Y%m%d_%H%M%S')
    runs_log_path=comm_args.runs_log_path[:-4]+"_"+cur_time+".txt"
    task_names = comm_args.tasks if comm_args.tasks != ["all"] else list(TASKS.keys())
    print("=================================")
    print(task_names)
    print("=================================")
    task_names_all_num=len(task_names)
    step=MAX_PROCESS_NUM
    processes=[]
    for i in range(0,task_names_all_num,step):
        start=i
        end=min(i+step,task_names_all_num)
        print("start: {} -- end: {} ".format(start,end))
        task_chunk=task_names[start:end]
        for task in task_chunk:
            p = multiprocessing.Process(target=one_task_eval, args=(task,runs_log_path,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

if __name__ == '__main__':

    multi_process_task_run()
