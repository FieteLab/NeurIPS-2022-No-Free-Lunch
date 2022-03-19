import numpy as np

def map_from_str(map_type):
    if map_type.lower() == "pls":
        from mec_hpc_investigations.neural_mappers import PLSNeuralMap
        return PLSNeuralMap
    elif map_type.lower() == "percentile":
        from mec_hpc_investigations.neural_mappers import PercentileNeuralMap
        return PercentileNeuralMap
    elif map_type.lower() == "factored":
        from mec_hpc_investigations.neural_mappers import FactoredNeuralMap
        return FactoredNeuralMap
    elif map_type.lower() == "identity":
        from mec_hpc_investigations.neural_mappers import IdentityNeuralMap
        return IdentityNeuralMap
    elif map_type.lower() == "sklinear":
        from mec_hpc_investigations.neural_mappers import SKLinearNeuralMap
        return SKLinearNeuralMap
    else:
        raise ValueError(f"{map_type.lower()} is not supported.")
