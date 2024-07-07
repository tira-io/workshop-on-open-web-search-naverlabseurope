# Retrieval with Splade

You can run the docker image via (please install `tira` via `pip3 install tira`, Docker, and Python >= 3.7 on your machine): 

```
tira-run \
	--input-dataset reneuir-2024/dl-top-10-docs-20240701-training \
	--command '/splade-retrieval-from-index.py --input $inputDataset --output $outputDir' \
	--image mam10eks/splade_tira:0.0.1-retrieval
```

## Development

You can build the Docker image via:

```
docker build -t mam10eks/splade_tira:0.0.1-retrieval .
```

To publish the image to dockerhub, run:

```
docker push mam10eks/splade_tira:0.0.1-retrieval
```
