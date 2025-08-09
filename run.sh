python -u train.py --cuda 0 --lr 0.001 --mu 0.0001 --eval_freq 100 --dim 200 --dataset DB15K --epochs 2000 --alpha_s 1e-5 --alpha_t 1e-5 --alpha_i 1e-5 --alpha_conf 1e-3 --alpha_cl 5e-5  > db15k.txt;
python -u train.py --cuda 0 --lr 0.001 --mu 0.0001 --eval_freq 100 --dim 200 --dataset MKG-W --epochs 2000 --alpha_s 1e-4 --alpha_t 1e-4 --alpha_i 1e-4 --alpha_conf 1e-4 --alpha_cl 1e-4  > mkgw.txt;
python -u train.py --cuda 0 --lr 0.001 --mu 0.0001 --eval_freq 100 --dim 200 --dataset MKG-Y --epochs 2000 --alpha_s 1e-3 --alpha_t 1e-3 --alpha_i 1e-3 --alpha_conf 1e-3 --alpha_cl 1e-3  > mkgy.txt;

