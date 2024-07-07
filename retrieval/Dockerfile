#docker build -t mam10eks/splade_tira:0.0.1-retrieval .
FROM mam10eks/splade_tira:0.0.1

RUN pip3 uninstall -y tira \
	&& pip3 install tira

ADD splade_retrieval.py splade_index.py splade-retrieval-from-index.py /

ENTRYPOINT [ "python3", "/splade_index.py", "--input", "$inputDataset", "--output", "$outputDir" ]
