import tensorflow as tf

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import *

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn
import os
from optimizers.gan import GraphGANOptimizer
batch_dim = 32
la = 1
dropout = 0
n_critic = 5
n_samples = 20000
z_dim = 32
epochs = 150
f_mapping=True
save_every = None
data = SparseMolecularDataset()
data.load('data/zinc_nodes.sparsedataset')
if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','first')):
    os.mkdir(os.path.join('summaries','first'))

direcs="logs/model_gan_zinc_16"
import pickle
imported_graph = tf.train.import_meta_graph(direcs+'/model.ckpt.meta')

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
steps = (len(data) // batch_dim)
if not os.path.exists(direcs):
    os.makedirs(direcs)

def test_feed_dict(model, optimizer, batch_dim):
    mols, _, _, a, x, _, _, _, _ = data.next_test_batch()
    embeddings = model.sample_z(a.shape[0])
    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(mols)

    feed_dict = {
                 
                 model.embeddings: embeddings}
    return feed_dict
model = GraphGANModel(data.vertexes,
                      data.bond_num_types,
                      data.atom_num_types,
                      z_dim,
                      decoder_units=(128, 256, 512),
                      discriminator_units=((64, 32), 128, (128, 64)),
                      decoder=decoder_adj,
                      discriminator=encoder_rgcn,
                      soft_gumbel_softmax=True,
                      hard_gumbel_softmax=False,
                      batch_discriminator=True)


with tf.Graph().as_default(), tf.Session() as session:
    saver = tf.train.import_meta_graph(direcs+'/model.ckpt.meta')
    saver.restore(session, direcs+'/model.ckpt')
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
  #  print("type of feeder:",type(test_feed_dict(batch_dim)))
   # print(test_feed_dict(batch_dim))


    sampled_embeddings = model.sample_z(20000)
    edges_out=tf.get_default_graph().get_tensor_by_name("outputs/one_hot_2:0")
    nodes_out=tf.get_default_graph().get_tensor_by_name("outputs/one_hot_3:0")
    print(edges_out,nodes_out)
    embeddings=tf.get_default_graph().get_tensor_by_name("Placeholder_2:0")
    print(embeddings)
    n, e = session.run([nodes_out,edges_out],
                               feed_dict={embeddings: sampled_embeddings})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    print(n.shape)
    print(n[0])
   # for i,j in zip(n,e):
   #     print("n:",i,"\ne:",j)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
    print(len(mols))
    smiles=[]
    for i in mols:
        try:
            i=i.GetMol()
            smi=Chem.MolToSmiles(i,isomericSmiles=True)
            smiles.append(smi)
        except:
            pass
    with open("smiles/zinc_gan_generated_smiles.txt","w") as f:
        for j in smiles:
            f.write("%s\n"%j)
 
