import logging

from .graph import Graph
from .adc import ADC_Feeder

__feeder = {
    'adc-xsub': ADC_Feeder,
}

__shape = {
    'adc-xsub': [3,6,300,23,1],
}

__class = {
    'adc-xsub': 18,
}

def create(dataset, path, preprocess=False, **kwargs):
    if dataset not in __class.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    graph = Graph(dataset)
    feeder_name = 'adc-xsub'
    kwargs.update({
        'path': '{}/{}'.format(path, dataset.replace('-', '/')),
        'data_shape': __shape[feeder_name],
        'connect_joint': graph.connect_joint,
        # 'debug': debug,
    })
    feeders = {
        'train': __feeder[feeder_name]('train', **kwargs),
        'eval' : __feeder[feeder_name]('eval', **kwargs),
    }
    # if 'adc' in dataset:
    #     feeders['adc'] = NTU_Location_Feeder(__shape[feeder_name])
    return feeders, __shape[feeder_name], __class[dataset], graph.A, graph.parts