
pip install -r requirements.txt

# wandb login

mkdir -p data_pairs

scp ai@46.245.80.20:/dataset.tar.xz .

tar -xJf dataset.tar.xz
cp dataset/* data_pairs
rm -r dataset.tar.xz
