# test of build programs

# some referred implementations
# https://github.com/yang-song/score_sde_pytorch/tree/main
# https://github.com/yang-song/score_inverse_problems/tree/main

# https://github.com/chenhu96/Self-Supervised-MRI-Reconstruction
# https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/tree/master






import run
from absl import app
from absl import flags

FLAGS = flags.FLAGS

# adding flag name and values
flags.DEFINE_string('workdir',None, "Work directory.")
flags.DEFINE_enum("mode", 'train', ["train","run"],"running mode")



def main(argv):
    print(FLAGS.workdir)
    print(FLAGS.mode)
    run.train()



if __name__=='__main__':
    app.run(main)