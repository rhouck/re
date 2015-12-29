FROM rothnic/anaconda-notebook  

RUN /home/condauser/anaconda3/bin/conda install basemap Quandl graphviz --yes
RUN /home/condauser/anaconda3/bin/conda update scikit-learn --yes
RUN /home/condauser/anaconda3/bin/pip install pykalman pydot2

CMD $PY3PATH/ipython notebook