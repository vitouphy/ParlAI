#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Daily Dialog https://arxiv.org/abs/1710.03957.

Original data is copyright by the owners of the paper, and free for use in research.

Every conversation contains entries with special fields (see the paper):

- emotion
- act_type
- topic

This teacher plays both sides of the conversation, once acting as Speaker 1, and
once acting as Speaker 2.
"""

import os
import json
import random
from parlai.core.teachers import FixedDialogTeacher, FbDialogTeacher
from parlai.core.params import ParlaiParser
from .build import build


START_ENTRY = {'text': '__SILENCE__', 'emotion': 'no_emotion', 'act': 'no_act'}

class Convai2Teacher(FixedDialogTeacher):

    @staticmethod
    def add_cmdline_args(parser):
        """
        Add command line args specific to this agent.
        """
        parser = parser.add_argument_group('DailyDialog Arguments')
        parser.add_argument(
            '-dhs',
            '--data_history_size',
            type=int,
            default=5,
            help='number of context utterances',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        if shared:
            self.data = shared['data']
        else:
            build(opt)
            fold = opt.get('datatype', 'train').split(':')[0]
            self._setup_data(fold)

        self.num_exs = sum(len(d['dialogue']) for d in self.data)
        self.start_token =  "__SOU__"
        self.speaker_a = "__speaker_A__"
        self.speaker_b = "__speaker_B__"
        self.end_token = "__EOU__"

        # we learn from both sides of every conversation
        self.num_eps = 2 * len(self.data)
        self.reset()

    def get_history_size(self):
        if self.history_size == -1:
            return random.randint(1, 5)
        return self.history_size

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, fold):
        self.data = []
        fpath = os.path.join(self.opt['datapath'], 'dailydialog', fold + '.json')
        with open(fpath) as f:
            for line in f:
                self.data.append(json.loads(line))

    def get(self, episode_idx, entry_idx=0):
        # Sometimes we're speaker 1 and sometimes we're speaker 2
        speaker_id = episode_idx % 2
        full_eps = self.data[episode_idx // 2]

        entries = [START_ENTRY] + full_eps['dialogue']
        their_turn = entries[speaker_id + 2 * entry_idx]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]

        episode_done = 2 * entry_idx + speaker_id + 1 >= len(full_eps['dialogue']) - 1

        action = {
            'topic': full_eps['topic'],
            'text': their_turn['text'],
            'emotion': their_turn['emotion'],
            'act_type': their_turn['act'],
            'labels': [my_turn['text']],
            'episode_done': episode_done,
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

class NoStartTeacher(Convai2Teacher):
    """
    Same as default teacher, but it doesn't contain __SILENCE__ entries.

    If we are the first speaker, then the first utterance is skipped.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        # Calculate the correct number of examples.
        self.num_exs = sum(len(d['dialogue']) - 1 for d in self.data)

        # Store all episodes separately, so we can deal with 2-turn dialogs.
        self.all_eps = self.data + [d for d in self.data if len(d['dialogue']) > 2]
        self.num_eps = len(self.all_eps)

    def get(self, episode_idx, entry_idx=0):
        full_eps = self.all_eps[episode_idx]
        entries = full_eps['dialogue']

        # Sometimes we're speaker 1 and sometimes we're speaker 2.
        # We can't be speaker 1 if dialog has only 2 turns.
        speaker_id = int(episode_idx >= len(self.data))

        their_turn = entries[speaker_id + 2 * entry_idx]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]
        episode_done = 2 * entry_idx + speaker_id + 1 >= len(entries) - 2

        action = {
            'topic': full_eps['topic'],
            'text': their_turn['text'],
            'emotion': their_turn['emotion'],
            'act_type': their_turn['act'],
            'labels': [my_turn['text']],
            'episode_done': episode_done,
        }
        return action

class ContextTeacher(Convai2Teacher):
    """
    Same as default teacher, but it doesn't contain __SILENCE__ entries.

    If we are the first speaker, then the first utterance is skipped.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        # Calculate the correct number of examples.
        self.num_exs = sum(len(d['dialogue']) - 1 for d in self.data)

        # Store all episodes separately, so we can deal with 2-turn dialogs.
        self.all_eps = self.data + [d for d in self.data if len(d['dialogue']) > 2]
        self.num_eps = len(self.all_eps)
        self.history_size = opt.get('data_history_size')
        
    
    def get(self, episode_idx, entry_idx=0):
        full_eps = self.all_eps[episode_idx]
        entries = full_eps['dialogue']
        history_size = self.get_history_size()
        # Sometimes we're speaker 1 and sometimes we're speaker 2.
        # We can't be speaker 1 if dialog has only 2 turns.
        
        speaker_id = int(episode_idx >= len(self.data))
        end_turn_idx = speaker_id + 2 * entry_idx
        start_turn_idx = max(0, end_turn_idx - history_size + 1)
        their_turns = entries[start_turn_idx : end_turn_idx+1]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]
        episode_done = 2 * entry_idx + speaker_id + 1 >= len(entries) - 2

        context_emotions = []
        context_acts = []
        context_texts = []

        for idx, turn in enumerate(their_turns):
            context_emotions.append(turn['emotion'])
            context_acts.append(turn['act'])
            context_texts.append(self.start_token)

            if ((len(their_turns) % 2 == 0) + idx) % 2 == 0:
                speaker_label = self.speaker_b 
            else: 
                speaker_label = self.speaker_a
                
            context_texts.append(speaker_label)
            context_texts.append(turn['text'])
            context_texts.append(self.end_token)

        context_text = " ".join(context_texts)

        action = {
            'topic': full_eps['topic'],
            'text': context_text,
            'emotion': context_emotions,
            'act_type': context_acts,
            'labels': [my_turn['text']],
            'labels_emotion': my_turn['emotion'],
            'labels_act': my_turn['act'],
            'episode_done': episode_done,
        }

        return action

class RerankTeacher(Convai2Teacher):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        # Calculate the correct number of examples.
        self.num_exs = sum(len(d['dialogue']) - 1 for d in self.data)

        # Store all episodes separately, so we can deal with 2-turn dialogs.
        self.all_eps = self.data + [d for d in self.data if len(d['dialogue']) > 2]
        self.num_eps = len(self.all_eps)
        self.history_size = opt.get('data_history_size')
        self.candidates = self.load_candidates(self.get_data_folder())

    def get_data_folder(self):
        dir_path = os.path.join(self.opt['datapath'], self.opt['task'].split(':')[0])
        datatype = self.opt['datatype'].split(':')[0]
        cand_path = os.path.join(dir_path, "cands_{}:ordered:stream.txt".format(datatype))
        return cand_path

    def load_candidates(self, cand_path):
        candidates = []
        with open(cand_path) as file:
            for line in file:
                candidates.append(line.strip())
            return candidates

    def get_candidates(self, groundtruth, num_cands=10, include_gt=True):

        # if we select all, just remove ONE grouthtruth from there
        if num_cands < 0:
            candidates = self.candidates.copy()
            match_indices = []
            for idx, cand in enumerate(candidates):
                if cand.strip() == groundtruth.strip():
                    match_indices.append(idx)
            candidates.pop(match_indices[0])
            if include_gt:
                candidates.append(groundtruth)
            return candidates
            

        # case that num_cands > 0
        total_cands = len(self.candidates)
        candidates = []

        # get 10 candidates. One groundtruth + 9 fake
        while len(candidates) < num_cands-1:
            idx = random.randint(0, total_cands-1)
            if self.candidates[idx] == groundtruth:
                continue
            candidates.append(self.candidates[idx])

        candidates.append(groundtruth)
        random.shuffle(candidates)
        return candidates
        

    def get(self, episode_idx, entry_idx=0):
        full_eps = self.all_eps[episode_idx]
        entries = full_eps['dialogue']
        history_size = self.get_history_size()
        # Sometimes we're speaker 1 and sometimes we're speaker 2.
        # We can't be speaker 1 if dialog has only 2 turns.
        
        speaker_id = int(episode_idx >= len(self.data))
        end_turn_idx = speaker_id + 2 * entry_idx
        start_turn_idx = max(0, end_turn_idx - history_size + 1)
        their_turns = entries[start_turn_idx : end_turn_idx+1]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]
        episode_done = 2 * entry_idx + speaker_id + 1 >= len(entries) - 2

        context_emotions = []
        context_acts = []
        context_texts = []

        datatype = self.opt['datatype'].split(':')[0]
        if datatype in ['train', 'valid']:
            candidates = self.get_candidates(my_turn['text'], num_cands=10, include_gt=True)
        elif datatype == 'test':
            candidates = self.get_candidates(my_turn['text'], num_cands=10, include_gt=True)

        for idx, turn in enumerate(their_turns):
            context_emotions.append(turn['emotion'])
            context_acts.append(turn['act'])
            context_texts.append(self.start_token)

            if ((len(their_turns) % 2 == 0) + idx) % 2 == 0:
                speaker_label = self.speaker_b 
            else: 
                speaker_label = self.speaker_a

            context_texts.append(speaker_label)
            context_texts.append(turn['text'])
            context_texts.append(self.end_token)

        context_text = " ".join(context_texts)

        action = {
            'topic': full_eps['topic'],
            'text': context_text,
            'emotion': context_emotions,
            'act_type': context_acts,
            'labels': [my_turn['text']],
            'label_candidates': candidates,
            'labels_emotion': my_turn['emotion'],
            'labels_act': my_turn['act'],
            'episode_done': episode_done,
        }

        return action
    
class DefaultTeacher(Convai2Teacher):
    pass
