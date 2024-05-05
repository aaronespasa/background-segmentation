#!/bin/bash

mkdir -p data

unzip -d data matting-human-small-dataset.zip

if [ $? -eq 0 ]; then
    echo "El dataset ha sido descomprimido correctamente 🎉"
else
    echo "Error: Failed to unzip the dataset."
fi

rm matting-human-small-dataset.zip
