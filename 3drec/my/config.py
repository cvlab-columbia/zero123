from typing import List, Union
from copy import deepcopy
from collections import namedtuple
from pathlib import Path
import argparse
from argparse import RawDescriptionHelpFormatter
import yaml
from pydantic import BaseModel as _Base


class BaseConf(_Base):
    class Config:
        validate_all = True
        allow_mutation = True
        extra = "ignore"


def SingleOrList(inner_type):
    return Union[inner_type, List[inner_type]]


def optional_load_config(fname="config.yml"):
    cfg = {}
    conf_fname = Path.cwd() / fname
    if conf_fname.is_file():
        with conf_fname.open("r") as f:
            raw = f.read()
            print("loaded config\n ")
            print(raw)  # yaml raw itself is well formatted
            cfg = yaml.safe_load(raw)
    return cfg


def write_full_config(cfg_obj, fname="full_config.yml"):
    cfg = cfg_obj.dict()
    cfg = _dict_to_yaml(cfg)
    print(f"\n--- full config ---\n\n{cfg}\n")
    with (Path.cwd() / fname).open("w") as f:
        f.write(cfg)


def argparse_cfg_template(curr_cfgs):
    parser = argparse.ArgumentParser(
        description='Manual spec of configs',
        epilog=f'curr cfgs:\n\n{_dict_to_yaml(curr_cfgs)}',
        formatter_class=RawDescriptionHelpFormatter
    )
    _, args = parser.parse_known_args()
    clauses = []
    for i in range(0, len(args), 2):
        assert args[i][:2] == "--", "please start args with --"
        clauses.append({args[i][2:]: args[i+1]})
    print(f"cmdline clauses: {clauses}")

    maker = ConfigMaker(curr_cfgs)
    for clu in clauses:
        maker.execute_clause(clu)

    final = maker.state.copy()
    return final


def _dict_to_yaml(arg):
    return yaml.safe_dump(arg, sort_keys=False, allow_unicode=True)


def dispatch(module):
    cfg = optional_load_config()
    cfg = module(**cfg).dict()

    cfg = argparse_cfg_template(cfg)  # cmdline takes priority
    mod = module(**cfg)

    write_full_config(mod)

    mod.run()


# below are some support tools


class ConfigMaker():
    CMD = namedtuple('cmd', field_names=['sub', 'verb', 'objs'])
    VERBS = ('add', 'replace', 'del')

    def __init__(self, base_node):
        self.state = base_node
        self.clauses = []

    def clone(self):
        return deepcopy(self)

    def execute_clause(self, raw_clause):
        cls = self.__class__
        assert isinstance(raw_clause, (str, dict))
        if isinstance(raw_clause, dict):
            assert len(raw_clause) == 1, \
                "a clause can only have 1 statement: {} clauses in {}".format(
                    len(raw_clause), raw_clause
            )
            cmd = list(raw_clause.keys())[0]
            arg = raw_clause[cmd]
        else:
            cmd = raw_clause
            arg = None
        cmd = self.parse_clause_cmd(cmd)
        tracer = NodeTracer(self.state)
        tracer.advance_pointer(path=cmd.sub)
        if cmd.verb == cls.VERBS[0]:
            tracer.add(cmd.objs, arg)
        elif cmd.verb == cls.VERBS[1]:
            tracer.replace(cmd.objs, arg)
        elif cmd.verb == cls.VERBS[2]:
            assert isinstance(raw_clause, str)
            tracer.delete(cmd.objs)
        self.state = tracer.state

    @classmethod
    def parse_clause_cmd(cls, input):
        """
        Args:
            input: a string to be parsed
        1. First test whether a verb is present
        2. If not present, then str is a single subject, and verb is replace
           This is a syntactical sugar that makes writing config easy
        3. If a verb is found, whatever comes before is a subject, and after the
           objects.
        4. Handle the edge cases properly. Below are expected parse outputs
        input       sub     verb        obj
        --- No verb
        ''          ''      replace     []
        'a.b'       'a.b'   replace     []
        'add'       ''      add         []
        'P Q' err: 2 subjects
        --- Verb present
        'T add'     'T'     add         []
        'T del a b' 'T'     del         [a, b]
        'P Q add a' err: 2 subjects
        'P add del b' err: 2 verbs
        """
        assert isinstance(input, str)
        input = input.split()
        objs = []
        sub = ''
        verb, verb_inx = cls.scan_for_verb(input)
        if verb is None:
            assert len(input) <= 1, "no verb present; more than 1 subject: {}"\
                .format(input)
            sub = input[0] if len(input) == 1 else ''
            verb = cls.VERBS[1]
        else:
            assert not verb_inx > 1, 'verb {} at inx {}; more than 1 subject in: {}'\
                .format(verb, verb_inx, input)
            sub = input[0] if verb_inx == 1 else ''
            objs = input[verb_inx + 1:]
        cmd = cls.CMD(sub=sub, verb=verb, objs=objs)
        return cmd

    @classmethod
    def scan_for_verb(cls, input_list):
        assert isinstance(input_list, list)
        counts = [ input_list.count(v) for v in cls.VERBS ]
        presence = [ cnt > 0 for cnt in counts ]
        if sum(presence) == 0:
            return None, -1
        elif sum(presence) > 1:
            raise ValueError("multiple verbs discovered in {}".format(input_list))

        if max(counts) > 1:
            raise ValueError("verbs repeated in cmd: {}".format(input_list))
        # by now, there is 1 verb that has occurred exactly 1 time
        verb = cls.VERBS[presence.index(1)]
        inx = input_list.index(verb)
        return verb, inx


class NodeTracer():
    def __init__(self, src_node):
        """
        A src node can be either a list or dict
        """
        assert isinstance(src_node, (list, dict))

        # these are movable pointers
        self.child_token = "_"  # init token can be anything
        self.parent = {self.child_token: src_node}

        # these are permanent pointers at the root
        self.root_child_token = self.child_token
        self.root = self.parent

    @property
    def state(self):
        return self.root[self.root_child_token]

    @property
    def pointed(self):
        return self.parent[self.child_token]

    def advance_pointer(self, path):
        if len(path) == 0:
            return
        path_list = list(
            map(lambda x: int(x) if str.isdigit(x) else x, path.split('.'))
        )

        for i, token in enumerate(path_list):
            self.parent = self.pointed
            self.child_token = token
            try:
                self.pointed
            except (IndexError, KeyError):
                raise ValueError(
                    "During the tracing of {}, {}-th token '{}'"
                    " is not present in node {}".format(
                        path, i, self.child_token, self.state
                    )
                )

    def replace(self, objs, arg):
        assert len(objs) == 0
        val_type = type(self.parent[self.child_token])
        # this is such an unfortunate hack
        # turn everything to string, so that eval could work
        # some of the clauses come from cmdline, some from yaml files for sow.
        arg = str(arg)
        if val_type == str:
            pass
        else:
            arg = eval(arg)
            assert type(arg) == val_type, \
                f"require {val_type.__name__}, given {type(arg).__name__}"

        self.parent[self.child_token] = arg
