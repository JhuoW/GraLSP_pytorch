import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GraLSP.")
    
    parser.add_argument("--dataset_name",
                        nargs="?",
                        default="cora",  
                        help="The dataset to use, corresponds to a folder under data/")
    
    parser.add_argument("--path_length",
                        type=int,
                        default=10, 
                        help="The length of random_walks")

    parser.add_argument("--num_paths",
                        type=int,
                        default=100,  
                        help="The number of paths to use per node")

    parser.add_argument("--window_size",
                        type=int,
                        default=6,  
                        help="The window size to sample neighborhood")                    

    parser.add_argument("--batch_size",
                        type=int,
                        default=100,  
                        help="batch size")  

    parser.add_argument("--neg_size",
                        type=int,
                        default=8,  
                        help="neg_size")
    
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.002,  
                        help="learning rate")

    parser.add_argument("--embedding_dims",
                        type=int,
                        default=32,  
                        help="The size of each embedding")
    

    parser.add_argument("--epochs",
                        type=int,
                        default=1,  
                        help="Steps to train")
    
    parser.add_argument("--num_skips",
                        type=int,
                        default=5,  
                        help="how many samples to draw from a single walk")   

    parser.add_argument("--num_neighbor",
                        type=int,
                        default=20,  
                        help="How many neighbors to sample, for graphsage")  
    

    parser.add_argument("--hidden_dim",
                        type=int,
                        default=100,  
                        help="The size of hidden dimension, for graphsage")  

    parser.add_argument("--walk_dim",
                        type=int,
                        default=30,  
                        help="The size of embeddings for anonym. walks.")  
    
    
    parser.add_argument("--anonym_walk_len",
                        type=int,
                        default=8,  
                        help="The length of each anonymous walk, 4 or 5")     

    parser.add_argument("--walk_loss_lambda",
                        type=float,
                        default=0.1,  
                        help="Weight of loss focusing on anonym walk similarity") 

    
    parser.add_argument("--linkpred_ratio",
                        type=float,
                        default=0.1,  
                        help="The ratio of edges being removed for link prediction") 

    parser.add_argument("--p",
                        type=float,
                        default=0.25,  
                        help="return parameter for node2vec walk")

    parser.add_argument("--q",
                        type=float,
                        default=1,  
                        help="out parameter for node2vec walk") 

    parser.add_argument("--inductive",
                        type=int,
                        default=0,  
                        help="whether to do inductive inference")

    parser.add_argument("--inductive_model_epoch",
                        type=int,
                        default=None,  
                        help="the epoch of the saved model")

    parser.add_argument("--inductive_model_name",
                        type=str,
                        default=None,  
                        help="the path towards the loaded model")

    parser.add_argument("--save_path",
                        type = str,
                        default="embeddings/GraLSP")

    return parser.parse_args()
