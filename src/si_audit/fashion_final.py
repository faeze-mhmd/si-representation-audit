                                                                

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .fashion_readouts import fit_readout
from .final_evaluation import require


def claim_once(directory, provenance):
                                                                            
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'evaluation_started.json').open('x',encoding='utf-8',newline='\n') as handle:
        json.dump({'started_at_utc':datetime.now(timezone.utc).isoformat(),**provenance},handle,indent=2)
        handle.write('\n')


def checked_readout(estimator,train_x,train_y,val_x,val_y,expected,expected_predictions):
                                                                                        
    replay,predicted=fit_readout(estimator,train_x,train_y,val_x,val_y)
    for key in ('validation','parameters','warnings','iterations'):
        require(replay[key]==expected[key],'validation readout drift: '+key)
    require(np.array_equal(predicted,expected_predictions),'validation prediction drift')
    return replay
