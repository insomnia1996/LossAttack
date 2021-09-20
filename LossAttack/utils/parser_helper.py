from LossAttack.models.word import WordParser
from LossAttack.models.char import CharParser
from LossAttack.models.word_tag import WordTagParser
from LossAttack.models.word_char import WordCharParser
from LossAttack.cmds.DEC.model import DECParser
import torch


def init_model(config):
    if config.mode == 'DEC':
        return DECParser(config)

def init_parser(config, embeddings):
    if config.input == 'word':
        return WordParser(config, embeddings)
    elif config.input == 'word_tag':
        return WordTagParser(config, embeddings)
    elif config.input == 'word_char':
        return WordCharParser(config, embeddings)
    else:
        return CharParser(config, embeddings)


def load_parser(fname):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    state = torch.load(fname, map_location=device)
    parser = init_parser(state['config'], state['embeddings'])
    parser.load_state_dict(state['state_dict'])
    parser.to(device)

    return parser


def is_chars_judger(model, tags = None, chars = None):
    if isinstance(model, CharParser) or isinstance(model, WordCharParser):
        return True if chars is None else chars
    else:
        return False if tags is None else tags
