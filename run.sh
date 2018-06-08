# good set of hyperparameters

#baseline
python zforcing_main.py --batch-size 10  --aux-weight-start 0.0 --kld-weight-start 0.0 --bwd-weight 0.0  --lr 5e-4 --aux-weight-end 0.000 --eval-interval 10

python zforcing_main.py --batch-size 10  --aux-weight-start 0.1 --kld-weight-start 0.1 --bwd-weight 1.0  --lr 1e-3 --aux-weight-end 0.001
