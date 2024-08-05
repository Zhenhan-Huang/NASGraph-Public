import os, sys
import logging

logger = logging.getLogger(__name__)


def setup_logging(outdir, fnm, verbose=False):
    logdir = os.path.join(outdir, 'logdir')
    os.makedirs(logdir, exist_ok=True)
    # remove all handlers associated with the root logger object
    # without this code snippet, log file might not be created
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    #logging.root.hanlders = []
    logging_config = {
        'level': logging.INFO,
        'format': '[%(filename)s: %(lineno)3d]: %(message)s',
    }

    if verbose:
        logging_config['stream'] = sys.stdout
    else:
        logging_config['filename'] = os.path.join(logdir, fnm)
        logging_config['filemode'] = 'a'
        print(f'create logging file "{os.path.join(logdir, fnm)}"')
    
    logging.basicConfig(**logging_config)


def save_arguments():
    cmds = ', '.join(sys.argv[1:])
    logger.info(f'Command line arguments: {cmds}')
