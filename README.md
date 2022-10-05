# Lipschitz-Bernoulli-Decoupling

This project is based on ULTRA (https://github.com/ULTR-Community/ULTRA). Please see its document for more details.

Our main code is in `ultra/learning_algorithm/lbd.py`.

## Load dataset

Please download datasets first, then run the following command:

```bash
bash ./example/Yahoo/offline_exp_pipeline.sh
bash ./example/Istella-S/offline_exp_pipeline.sh
```

## Run experiment

- The following commands show the experiment settings on $\eta=0.1$. To simulate other degrees of coupling level, please modify the `weight` attribute in `config/click_*_0.1.json` by multiplying them with $\eta\times 10$. For example, to simulate $\eta=0.5$, you can multiply the weights by 5.

- Yahoo!
```bash
python3 main.py \
    --data_dir=./Yahoo_letor/tmp_data/ \
    --model_dir=./tmp/model/ \
    --output_dir=./tmp/model/output/ \
    --setting_file=./config/setting_yahoo_0.1_DNN.json
```

- Istella-S
```bash
python3 main.py \
    --data_dir=./istella-s-letor/tmp_data/ \
    --model_dir=./tmp/model/ \
    --output_dir=./tmp/model/output/ \
    --setting_file=./config/setting_istella_0.1_DNN.json
```

## Citation

Please consider citing the following paper when using our code for your application.

```
@inproceedings{chen2022lbd,
  title={LBD: Decouple Relevance and Observation for Individual-Level Unbiased Learning to Rank},
  author={Mouxiang Chen and Chenghao Liu and Zemin Liu and Jianling Sun},
  booktitle={NeurIPS},
  year={2022}
}
```

