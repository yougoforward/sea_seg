from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .encnet import *

from .fcn import *

from .cfpn_gsf import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'encnet': get_encnet,

        'cfpn_gsf': get_cfpn_gsf,

    }
    return models[name.lower()](**kwargs)
