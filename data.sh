#!/bin/bash

mkdir -p data

unzip -d data matting-human-small-dataset.zip

if [ $? -eq 0 ]; then
    echo "El dataset ha sido descomprimido correctamente 🎉"
else
    echo "Error: Failed to unzip the dataset."
fi


rm matting-human-small-dataset.zip

mkdir -p data/test/original
mkdir -p data/test/matting

counter=1
for i in {1500..2159}; do
    mv data/validation/original/$i.jpg data/test/original/$counter.jpg
    mv data/validation/matting/$i.png data/test/matting/$counter.png
    ((counter++))
done

echo "Images moved to test folders successfully."
