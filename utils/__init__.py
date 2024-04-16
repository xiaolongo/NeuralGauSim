import utils.load_data as load_data
import utils.logger as logger
import utils.knn as knn

glorot = load_data.glorot
zeros = load_data.zeros
load_data = load_data.load_data
Logger = logger.Logger
build_knn_graph = knn.build_knn_graph
knn_fast = knn.knn_fast
heat_knn_fast = knn.heat_knn_fast
