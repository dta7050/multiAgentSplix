''' This file is the entry point to the project. It takes in the user arguments,
processes it and then calls the appropriate functions accordingly '''

import argparse

import ActorCritic
import AsynchronousQ
import NewAlgo
import Constants


def main():
    """
    Calling this function runs the entire program. Depending on the arguments
    passed to it, the agents can be either trained or simulated using either
    an AsynchronousQ or Actor Critic algorithm. Additionally, previously
    trained agents can be loaded to be further trained or to evaluate how they
    perform.
    :return:
    """
    parser = argparse.ArgumentParser()  # used to add arguments that can be passed to the function
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=str,
                        choices=['train', 'simulate'],
                        required=True,
                        help="""Sets mode of execution.
                                If train, then the RL agents are trained using
                                specified algorithm and number of time_steps.
                                If simulate, then an episode is simulated using
                                pre-trained agents.""")
    parser.add_argument('--train_time_steps', type=int, default=10000,
                        help="The number of time steps to run training for. Requires: --mode=train.")
    parser.add_argument('--algorithm', type=str, choices=['asyncQ', 'newalgo', 'actorcritic'], required=True,
                        help="The algorithm to be used for training.")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help="The directory to store all checkpoints during training. Requires: --mode=train.")
    parser.add_argument('--checkpoint_frequency', type=int, default=500,
                        help="The number of iterations after which to periodically checkpoint the weights. Requires: --mode=train.")

    parser.add_argument('--play', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Setting this allows user to play the game alongside RL agents. Requires: --mode=simulate.")
    parser.add_argument('--trained_dir', type=str, default='checkpoints',
                        help="Directory to load checkpoints/numpy files from.")
    parser.add_argument('--trained_ckpt_index', type=int, default=-1,
                        help="The checkpoint number that refers the file to be loaded.")

    parser.add_argument('--hidden_units', type=int, default=20,
                        help="Size of the hidden layer of neural network. Requires: --algorithm=asyncQ")
    parser.add_argument('--threads', type=int, default=4,
                        help="The number of threads to run training on. Requires: --algorithm=asyncQ")

    args = parser.parse_args()

    if args.mode == 'train':  # train the agents
        load = (args.trained_ckpt_index != -1)  # if the checkpoint given is -1 (default) loading cannot be done
        if args.algorithm == 'asyncQ':
            AsynchronousQ.train(max_time_steps=args.train_time_steps, reward=1, penalty=-10,
                                size_of_hidden_layer=args.hidden_units, num_threads=args.threads,
                                checkpointFrequency=args.checkpoint_frequency, checkpoint_dir=args.checkpoint_dir,
                                load=load, load_dir=args.trained_dir, load_time_step=args.trained_ckpt_index)
        elif args.algorithm == 'newalgo':
            NewAlgo.train(max_time_steps=args.train_time_steps, reward=1, penalty=-10,
                         size_of_hidden_layer=args.hidden_units, num_threads=args.threads,
                         checkpointFrequency=args.checkpoint_frequency, checkpoint_dir=args.checkpoint_dir,
                         load=load, load_dir=args.trained_dir, load_time_step=args.trained_ckpt_index)
        else:
            ActorCritic.train(args.train_time_steps, checkpointFrequency=args.checkpoint_frequency,
                              checkpoint_dir=args.checkpoint_dir, load=load, load_dir=args.trained_dir,
                              load_time_step=args.trained_ckpt_index)
        print("Training complete.")
    else:
        if args.algorithm == 'asyncQ':  # simulate a game on trained agents
            AsynchronousQ.graphical_inference(args.hidden_units, load_dir=args.trained_dir,
                                              load_time_step=args.trained_ckpt_index,
                                              play=args.play, scalingFactor=9)
        elif args.algorithm == 'newalgo':
            NewAlgo.graphical_inference(args.hidden_units, load_dir=args.trained_dir,
                                        load_time_step=args.trained_ckpt_index,
                                        play=args.play, scalingFactor=9)
        else:
            ActorCritic.graphical_inference(load_dir=args.trained_dir, load_time_step=args.trained_ckpt_index,
                                            play=args.play, scalingFactor=9)
        print("Inference complete.")


if __name__ == '__main__':
    main()
