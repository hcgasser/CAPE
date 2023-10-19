import os

from kit.bioinf.fasta import translate_fasta


def pack_to_source_task_step(pack):
    if not pack.startswith('data') and not pack.startswith('support'):
        tmp = pack.split('.')
        if tmp[1] == 'generate':
            return (tmp[0], tmp[1], tmp[1])
        return (tmp[0], tmp[1], tmp[1]) if len(tmp) == 2 else tmp
    elif pack.startswith('support'):
        return 'support', 'support', 'support'
    elif pack.startswith('data'):
        return 'natural', 'natural', 'natural'
    
