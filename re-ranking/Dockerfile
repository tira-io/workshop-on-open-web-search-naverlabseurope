#docker build -t mam10eks/splade_tira:0.0.1-re-ranking .
FROM mam10eks/splade_tira:0.0.1

ADD splade_re_ranking.py /splade_re_ranking.py

ENTRYPOINT [ "python3", "/splade_re_ranking.py" ]
