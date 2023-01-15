#! /bin/bash
if ! [ "$(pip freeze | grep gdown)" ]; then
    echo "Install gdwon..."
    pip install gdown 
fi  
gdown https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8
tar zxvf 2016-01.tgz
mkdir ./data/iwslt
mv 2016-01/texts/en/de/en-de.tgz ./data/iwslt
mv 2016-01/texts/de/en/de-en.tgz ./data/iwslt
rm -r 2016-01 2016-01.tgz
