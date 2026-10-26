                                                                         

                                             
                                                                          
   

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SeparationResult:
                                                                    

                                                                            
                                                                             
                                                                             
                                                                           
       

    score: float
    n_samples: int
    n_features: int
    nearest_neighbor_indices: NDArray[np.int64]
    nearest_neighbor_ids: NDArray
    nearest_distances: NDArray[np.float64]
    same_label: NDArray[np.bool_]
    tie_fraction: float
    duplicate_vector_fraction: float


def separation_index(
    representations: ArrayLike,
    labels: ArrayLike,
    *,
    sample_ids: ArrayLike | None = None,
    batch_size: int = 64,
) -> SeparationResult:
                                                                            

           
                                                                            
                                                                              
                                                                              
                                                   
                                                                             
                                                                              
                                                                          
                                                                             

                                                                         
                                                                      
                                                                           
                                                                             
                                                                           
                                                                             

                                                                         
                                                                          
                                                                           
                                                                 
       
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral):
        raise ValueError("batch_size must be a positive integer")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    raw = np.asarray(representations)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 1:
        raise ValueError("representations must have shape (N, D), N >= 2, D >= 1")
    if raw.dtype.kind not in "iuf":
        raise ValueError("representations must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("representations must be finite in float64")
    n_samples, n_features = values.shape

    targets = np.asarray(labels)
    if targets.ndim != 1 or targets.shape[0] != n_samples:
        raise ValueError("labels must be one-dimensional and match N")
    if targets.dtype.kind not in "biufUS":
        raise ValueError("labels must have a numeric, boolean or string dtype")
    if targets.dtype.kind in "US":
                                                                     
        original_labels = np.asarray(labels, dtype=object)
        expected_type = str if targets.dtype.kind == "U" else bytes
        if any(not isinstance(label, expected_type) for label in original_labels):
            raise ValueError("string labels must be homogeneous; encode mixed label types explicitly")
    if targets.dtype.kind in "iuf" and not np.isfinite(targets).all():
        raise ValueError("numeric labels must be finite")

    ids = np.arange(n_samples, dtype=np.int64) if sample_ids is None else np.asarray(sample_ids)
    if ids.ndim != 1 or ids.shape[0] != n_samples or ids.dtype.kind not in "iu":
        raise ValueError("sample_ids must be a one-dimensional integer array matching N")
    if np.unique(ids).size != n_samples:
        raise ValueError("sample_ids must be unique")

                                                                        
    reference_order = np.argsort(ids, kind="stable")
    reference_values = values[reference_order]
    self_columns = np.empty(n_samples, dtype=np.int64)
    self_columns[reference_order] = np.arange(n_samples)
    neighbors = np.empty(n_samples, dtype=np.int64)
    nearest_squared = np.empty(n_samples, dtype=np.float64)
    tied = np.empty(n_samples, dtype=np.bool_)

    for start in range(0, n_samples, int(batch_size)):
        stop = min(start + int(batch_size), n_samples)
        try:
            with np.errstate(over="raise", invalid="raise"):
                differences = values[start:stop, None, :] - reference_values[None, :, :]
                distances = np.einsum("bnd,bnd->bn", differences, differences)
        except FloatingPointError as error:
            raise ValueError("pairwise squared distances overflow float64; rescale inputs") from error
        if not np.isfinite(distances).all():
            raise ValueError("pairwise squared distances overflow float64; rescale inputs")

        rows = np.arange(stop - start)
        distances[rows, self_columns[start:stop]] = np.inf
        selected_columns = np.argmin(distances, axis=1)
        minima = distances[rows, selected_columns]
        neighbors[start:stop] = reference_order[selected_columns]
        nearest_squared[start:stop] = minima
        tied[start:stop] = np.count_nonzero(distances == minima[:, None], axis=1) > 1

    same_label = targets == targets[neighbors]
    _, inverse, counts = np.unique(values, axis=0, return_inverse=True, return_counts=True)
    return SeparationResult(
        score=float(np.mean(same_label)),
        n_samples=n_samples,
        n_features=n_features,
        nearest_neighbor_indices=neighbors,
        nearest_neighbor_ids=ids[neighbors],
        nearest_distances=np.sqrt(nearest_squared),
        same_label=same_label,
        tie_fraction=float(np.mean(tied)),
        duplicate_vector_fraction=float(np.mean(counts[inverse] > 1)),
    )
