    mv ../scripted_notebook/*.py ../
    mv ../notebook/*.ipynb ../
    jupytext --to py:percent ../*.ipynb
    mv ../*.py ../scripted_notebook/
    mv ../*.ipynb ../notebook/