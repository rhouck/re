FROM rothnic/anaconda-notebook  

RUN /home/condauser/anaconda3/bin/conda install basemap --yes

CMD $PY3PATH/ipython notebook