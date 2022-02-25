    mv ../scripted_notebook/*.py ../
    mv ../notebook/*.ipynb ../
    jupytext --to notebook --update ../*.py
    mv ../*.py ../scripted_notebook/
    mv ../*.ipynb ../notebook/