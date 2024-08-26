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
    parser.add_argument('--fed_flag', type=str, required=True, help="The SplitFed Learning flag.")
    parser.add_argument('--proto_flag', type=str, required=True, help="The Prototypical Contrastive Learning flag.")
    parser.add_argument('--self_kd_flag', type=str, required=True, help="The Self-Knowledge Distillation flag.")
    parser.add_argument('--con_flag', type=str, required=True, help='The Model Contrastive Learning flag')
    parser.add_argument('--model_name', type=str, required=True, help='The train model.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Tha path of dataset directory')
    parser.add_argument('--dataset_type', type=str, required=True, help='Tha type of dataset')
    parser.add_argument('--results_path', type=str, required=True, help="The path of results directory")
    parser.add_argument('--date', type=str, required=True, help="The date.")
    parser.add_argument('--queue_size', type=int, required=True, help="The queue size of prototype")
    parser.add_argument('--save_data', type=str, required=True, help="AccuracyやLossを保存するかどうか")

    args = parser.parse_args()
    args.fed_flag = str_to_bool(args.fed_flag)
    args.proto_flag = str_to_bool(args.proto_flag)
    args.con_flag = str_to_bool(args.con_flag)
    args.self_kd_flag = str_to_bool(args.self_kd_flag)
    args.save_data = str_to_bool(args.save_data)

    # MOONはSFLじゃないと実行できない
    if args.con_flag == True and args.fed_flag == False:
        raise Exception('fed_flag is False')

    return args

def fl_arg_parser():

    parser = argparse.ArgumentParser('Federated Learning Simulation')

    parser.add_argument('--port_number', type=int, required=True, help="The port number.")
    parser.add_argument('--seed', type=int, required=True, help="The seed number.")
    parser.add_argument('--num_clients', type=int, required=True, help="The number of clients.")
    parser.add_argument('--num_rounds', type=int, required=True, help="The number of rounds.")
    parser.add_argument('--num_epochs', type=int, required=True, help="The number of epochs.")
    parser.add_argument('--batch_size', type=int, required=True, help="The batch size.")
    parser.add_argument('--lr', type=float, required=True, help="The learning rate.")
    parser.add_argument('--momentum', type=float, required=True, help="The momentum.")
    parser.add_argument('--weight_decay', type=float, required=True, help="The weight decay.")
    parser.add_argument('--data_partition', type=int, required=True, help="The data partition.(IID, Non-IID, Dirichlet)")
    parser.add_argument('--model_name', type=str, required=True, help='The train model.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Tha path of dataset directory')
    parser.add_argument('--dataset_type', type=str, required=True, help='Tha type of dataset')
    parser.add_argument('--results_path', type=str, required=True, help="The path of results directory")
    parser.add_argument('--date', type=str, required=True, help="The date.")
    parser.add_argument('--save_data', type=str, required=True, help="AccuracyやLossを保存するかどうか")

    args = parser.parse_args()
    args.save_data = str_to_bool(args.save_data)

    return args

def str_to_bool(value):
    if isinstance(value, str):
        if value == 'True':
            return True
        elif value == 'False':
            return False
    else:
        raise ValueError(f"Cannot conver {value} to bool")