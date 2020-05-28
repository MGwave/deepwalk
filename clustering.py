from metrics import clustering_metrics
from sklearn.cluster import KMeans
from gensim.models import Word2Vec, KeyedVectors
from scipy.io import loadmat
import numpy
import pickle as pkl

def evaluate(emb,labels,K=2):
    kmeans = KMeans(K, random_state=0).fit(emb)
    predict_labels = kmeans.predict(emb)
    cm = clustering_metrics(labels, predict_labels)
    cm.evaluationClusterModelFromLabel()
    with open('/home/zmm/advGraph/nettack-master/ourDefense/clusterLabel/dw_labels_polblogs' ,'wb') as f:
        pkl.dump(predict_labels,f)
    
embeddings_file = 'polblogs.embeddings'
matfile = 'example_graphs/polblogs.mat'
mat = loadmat(matfile)
A = mat['network']
graph = A.A #1490*1490
    
# 1. Load Embeddings and labels
model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
vocab = [int(i) for i in list(model.wv.vocab.keys())]
vocab.sort()
emb  = numpy.asarray([model[str(node)] for node in vocab]) #1224*64
empty = [i for i in range(graph.shape[0]) if i not in vocab]# 266
labels_all = mat['group'].nonzero()[1] #(1490,)
labels = labels_all[vocab] #(1224,)

kmeans = KMeans(2, random_state=0).fit(emb)
predict_labels = kmeans.predict(emb)
cm = clustering_metrics(labels, predict_labels)
cm.evaluationClusterModelFromLabel()
predict_labels_all = numpy.ones(graph.shape[0])*(-1)
print(predict_labels_all)
print(predict_labels_all.shape)
print(predict_labels.shape)
predict_labels_all[vocab] = predict_labels
predict_labels_all = predict_labels_all.astype(int)
print(predict_labels_all)
with open('/home/zmm/advGraph/nettack-master/ourDefense/clusterLabel/dw_labels_polblogs' ,'wb') as f:
    pkl.dump(predict_labels_all,f)




# python example_graphs/scoring.py --emb example_graphs/polblogs.embeddings --network example_graphs/polblogs.mat --num-shuffle 10 --all
