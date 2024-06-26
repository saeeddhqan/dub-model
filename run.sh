
pip install -r requirements.txt

wandb login

scp ai@46.245.80.20:/home/ai/train_pairs.tar.xz .
tar -xJf train_pairs.tar.xz
mv dataset train_pairs
rm -r train_pairs.tar.xz

scp ai@46.245.80.20:/home/ai/test_pairs.tar.xz .
tar -xJf test_pairs.tar.xz
mv dataset_test test_pairs
rm -r test_pairs.tar.xz
