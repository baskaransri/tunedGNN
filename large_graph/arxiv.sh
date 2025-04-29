#gcn
#python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 512 --epochs 2000 --lr 0.0005 --runs 2 --local_layers 5 --bn --device 0 --res 
#python basic-logs.py --dataset ogbn-arxiv --hidden_channels 512 --epochs 2000 --lr 0.0005 --runs 1 --local_layers 2 --device 0 --in_dropout 0 --dropout 0
#python basic-logs2.py --dataset ogbn-arxiv --hidden_channels 512 --epochs 2000 --lr 0.0005 --runs 1 --local_layers 5 --bn --device 0
python basic-logs.py --dataset cora --hidden_channels 512 --epochs 20 --lr 0.0005 --runs 1 --local_layers 2  --device 0

#sage
#python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 256 --epochs 2000 --lr 0.0005 --runs 2 --local_layers 4 --bn --device 0 --res --sage

