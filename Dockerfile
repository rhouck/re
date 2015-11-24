    FROM rothnic/anaconda-notebook  

RUN /home/condauser/anaconda3/bin/conda install basemap Quandl --yes

CMD $PY3PATH/ipython notebook