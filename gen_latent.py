
import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn
import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.vae import GraphVAEModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.vae import GraphVAEOptimizer
batch_dim = 32
la = 1
kl_weight=0.5
n_critic = 5
metric = 'qed'
n_samples = 500
z_dim = 32
epochs = 200
save_every = None

data = SparseMolecularDataset()
data.load('data/cvae16_nodes.sparsedataset')

steps = (len(data) // batch_dim)
directory="logs/model_vae_idg_16"
import pickle
imported_graph = tf.train.import_meta_graph(directory+'/model.ckpt.meta')
def test_fetch_dict(model, optimizer):
    return {'loss VAE': optimizer.loss_VAE, "kl": optimizer.kl_weight}


def test_feed_dict(batch_dim):
    mols, _, _, a, x, _, F, _, _ = data.next_train_batch()

    feed_dict = {edges_labels: a,
                 nodes_labels: x}
    return feed_dict

# list all the tensors in the graph
#for tensor in tf.get_default_graph().get_operations():
#    if "encoder" in tensor.name:
#        print (tensor)
#
with tf.Graph().as_default(), tf.Session() as session:
    saver = tf.train.import_meta_graph(directory+'/model.ckpt.meta')
    saver.restore(session, directory+'/model.ckpt')
    op = session.graph.get_operations()
    listops=[m.name for m in op]
#    for v in tf.get_default_graph().as_graph_def().node:
#        if v.name[:12]=="encoder/cond":
#        if "encoder" in v.name: 
#            print(v.name)
    for n in tf.get_default_graph().as_graph_def().node:
        print (n.name)
  #  print([i for i in listops if "hard_gumbel_softmax" in i])
    graph = tf.get_default_graph()
    res1=tf.get_default_graph().get_tensor_by_name("encoder/latent_layer/Merge:0")

    edges_labels=tf.get_default_graph().get_tensor_by_name("Placeholder:0")
    nodes_labels=tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
   # node_features=tf.get_default_graph().get_tensor_by_name("Placeholder_2:0")
    print("feed_dict")
    print(test_feed_dict(batch_dim))
  #  print("type of feeder:",type(test_feed_dict(batch_dim)))
   # print(test_feed_dict(batch_dim))


    output=session.run(res1,feed_dict=test_feed_dict(batch_dim))
    import pandas as pd
    results=pd.DataFrame(output)
    print(results.shape)
    results.to_csv("smiles/vae_idg_latent_vector.csv")
            

    #list_of_tuples = [op.values() for op in graph.get_operations()]
    #for i in list_of_tuples:
    #    print(i)
    #output = session.run(latent_layer,feed_dict=test_feed_dict(batch_dim))
    #print(output)
    #all_vars = tf.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
