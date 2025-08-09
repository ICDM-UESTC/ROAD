python -u train.py --cuda 0 --lr 0.001 --eval_freq 100 --dim 200 --dataset DB15K --epochs 2000 --beta_s 1e-5 --beta_t 1e-5 --beta_i 1e-5 --lamda_conf 1e-3 --lamda_cl 5e-5  > db15k.txt;
python -u train.py --cuda 0 --lr 0.001 --eval_freq 100 --dim 200 --dataset MKG-W --epochs 2000 --beta_s 1e-4 --beta_t 1e-4 --beta_i 1e-4 --lamda_conf 1e-4 --lamda_cl 1e-4  > mkgw.txt;
python -u train.py --cuda 0 --lr 0.001  --eval_freq 100 --dim 200 --dataset MKG-Y --epochs 2000 --beta_s 1e-3 --beta_t 1e-3 --beta_i 1e-3 --lamda_conf 1e-3 --lamda_cl 1e-3  > mkgy.txt;

