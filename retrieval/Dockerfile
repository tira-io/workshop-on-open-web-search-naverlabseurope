#docker build -t mam10eks/splade_tira:0.0.1-retrieval .
FROM mam10eks/splade_tira:0.0.1

ADD splade_retrieval.py /splade_retrieval.py

ENTRYPOINT [ "python3", "/splade_retrieval.py", "--input", "$inputDataset", "--output", "$outputDir" ]
