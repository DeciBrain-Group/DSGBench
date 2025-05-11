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
"""
Base Prompt Handler @ 

"""

import os
import re
import copy
from civrealm.freeciv.utils.freeciv_logging import fc_logger

# print("os.getcwd(=========",os.getcwd())
PROMPT_ROOT_DIR = "./agent_manager/prompts/civ_prompt_collections/"
BASE_DIR = "base_prompts/"


class BasePromptHandler:
    """
    Base Prompt Handler

    Provides interface together with a basic implementation.

    The prompts can be in .txt format and can use `<% name %>` for certain
    replacement of variables, provided the variables are given.

    [TODO] How to implement `if` and `for` in the setup? I think importing
    python scripts could be a solution, but another processor must be written.
    """
    CONF_FNAME = "__settings__.conf"

    def __init__(self, prompt_prefix: str = BASE_DIR):
        """
        Initialize.

        Parameters
        ----------
        prompt_prefix: str, must be a folder in a relative form.
        """
        self.prompt_prefix = PROMPT_ROOT_DIR + self._ending_dir(prompt_prefix)
        if not os.path.exists(self.prompt_prefix):
            if os.path.exists(prompt_prefix):
                self.prompt_prefix = prompt_prefix
            else:
                raise Exception(
                    f"Prompt prefix dir `{prompt_prefix}` does not exist! " +
                    f"Use 'example' for {PROMPT_ROOT_DIR}example/," +
                    "or simply the full path.")
        self.templates = {}
        self.related_pclasses = [self.prompt_prefix]
        self._solve_dependency()
        for prefix in reversed(self.related_pclasses):
            self._load_prompt_templates(prefix)
        print("====BasePromptHandler====")

    @staticmethod
    def _ending_dir(path: str):
        """Complete '/' to the end of a dir-path."""
        return path + ("" if path[-1] == '/' else '/')

    @staticmethod
    def _regularize(key: str) -> str:
        # Maybe, more rules?
        return key.replace(".", "_")

    def _solve_dependency(self):
        """
        Solve the dependency of prompts.

        BFS on dependency graph.
        """
        pclass = [self.prompt_prefix]

        while len(pclass) > 0:
            current = pclass.pop(0)
            current = self._ending_dir(current)
            new_dependency = []
            with open(current + self.CONF_FNAME, "r",
                      encoding="utf-8") as filep:
                for line in filep:
                    if line.strip()[0] == "#":
                        continue
                    conf_kv = line.strip().split(":")
                    if conf_kv[0].strip().lower() == "parents":
                        new_dependency += conf_kv[1].split(",")
                        break
            for name in new_dependency:
                path = PROMPT_ROOT_DIR + self._ending_dir(name.strip())
                if path not in self.related_pclasses:
                    self.related_pclasses.append(path)
                    pclass.append(path)

    def _txt_parser(self, raw: str, template_name: str) -> callable:
        def parser(_raise_empty: bool = False, **kwargs) -> str:
            nonlocal raw, template_name, self
            variables = set(re.findall("(<%[ ]+(.*?)[ ]+%>)", raw))
            recursions = set(re.findall(r"(<\$[ ]+(.*?)[ ]+\$>)", raw))

            out = copy.deepcopy(raw)
            for pattern, key in variables:
                try:
                    out = out.replace(pattern, str(kwargs[key]))
                    # I decide not to use `kwargs.get(key, "")` in order to
                    # log the incidents when key is not provided in args.
                except KeyError as einfo:
                    fc_logger.error(f"Failed to provide key {key}" +
                                    f" in generating {template_name}.")
                    fc_logger.error(einfo)
                    print(f"Failed to provide key {key}" +
                          f" in generating {template_name}.")
                    if _raise_empty:
                        raise
                    out = out.replace(pattern, "")
            for pattern, key in recursions:
                try:
                    func_end = key.find("(")
                    if func_end == -1:
                        key = key + "()"
                        func_end = -2
                    if key[:func_end] == template_name:
                        print("Should raise")
                        raise Exception("Self-quoted!")
                    # replace = eval(
                    #     f"self.generate['{key[:func_end]}']({key[func_end:]})")
                    # print(self.templates.keys())
                    replace = eval("self." + key)

                    print(replace)
                    out = out.replace(pattern, replace)
                except Exception as einfo:
                    fc_logger.error(f"Failed to load submodule {key}" +
                                    f" in generating {template_name}.")
                    fc_logger.error(einfo)
                    print(f"Failed to load submodule {key}" +
                          f" in generating {template_name}.")
                    print(einfo)
                    if _raise_empty:
                        raise
                    out = out.replace(pattern, "")
            return out

        return parser

    def _load_prompt_templates(self, prefix: str):
        # load text-style templates
        try:
            files = list(os.walk(prefix))[0][2]

        except Exception as einfo:
            print("Error in loading prompt files.", einfo)
            raise einfo

        for fname in [
                fname[:-4] for fname in files
                if fname[-4:] == ".txt" and "#" not in fname
        ]:
            with open(prefix + fname + ".txt", "r", encoding="utf-8") as filep:
                raw = filep.read()
            key = self._regularize(fname)
            # self.templates[key] = self._txt_parser(raw, key)
            self.templates[fname] = self._txt_parser(raw, fname)
            # This should provide a runtime-object-level method management.
            setattr(self, key, self.templates[fname])

    def generate(self,
                 _prompt_key: str,
                 _raise_empty: bool = False,
                 **kwargs) -> str:
        """
        Parameters
        ----------
        _prompt_key: str, template prompt_key
        _raise_empty: bool, if True, raise any unprovided slots in template
        **kwargs : args

        Returns
        -------
        out : The prompt piece.
        """
        try:
            return self.templates[_prompt_key](**kwargs)
        except KeyError as einfo:
            fc_logger.error(
                f"Keyerror {_prompt_key}: No such template registered.")
            fc_logger.error(einfo)
            print(f"Keyerror {_prompt_key}: No such template registered.")
            raise

    # def add_prompt_prefix()
    # Maybe not a good idea?


def unit_test():
    """The unit test."""
    print(list(os.walk("."))[0])
    phandler = BasePromptHandler(PROMPT_ROOT_DIR + "test_prompts_01_settlers/")
    print(phandler.insist_json())
    return phandler.generate("test_prompt",
                             uid="112",
                             utype="Carrier",
                             uid2="312")


if __name__ == '__main__':
    print(unit_test())
