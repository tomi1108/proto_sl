import argparse

def arg_parser():

    parser = argparse.ArgumentParser("Split Learning Simulation")

    parser.add_argument('--port_number', type=int, required=True, help="The port number.")
    parser.add_argument('--seed', type=int, required=True, help="The seed number.")
    parser.add_argument('--num_clients', type=int, required=True, help="The number of clients.")
    parser.add_argument('--num_rounds', type=int, required=True, help="The number of rounds.")
    parser.add_argument('--num_epochs', type=int, required=True, help="The number of epochs.")
    parser.add_argument('--batch_size', type=int, required=True, help="The batch size.")
    parser.add_argument('--lr', type=float, required=True, help="The learning rate.")
    parser.add_argument('--momentum', type=float, required=True, help="The momentum.")
    parser.add_argument('--weight_decay', type=float, required=True, help="The weight decay.")
    parser.add_argument('--temperature', type=float, required=True, help="The temperature.")
    parser.add_argument('--output_size', type=int, required=True, help="The output size.")
    parser.add_argument('--data_partition', type=int, required=True, help="The data partition.(IID, Non-IID, Dirichlet)")
    parser.add_argument('--fed_flag', type=bool, required=True, help="The SplitFed Learning flag.")
    parser.add_argument('--proto_flag', type=bool, required=True, help="The Prototypical Contrastive Learning flag.")
    parser.add_argument('--self_kd_flag', type=bool, required=True, help="The Self-Knowledge Distillation flag.")

    return parser.parse_args()