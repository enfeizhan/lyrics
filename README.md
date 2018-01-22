# Lyrics generator
## Manage envrionment

Try to run this to create the name environment:
```
conda env create -f environment.yml
```
## Invoke environment
Run this to bring up the environment:
```
source activate py3
```
If ``activate`` not found, check if conda is in path.

## Start jupyter notebook
```
jupyter notebook
```
In jupyter notebook, click ``implement.ipynb``.
Copy the model file ``modelYYYYMMDD_X.h5``, ``ind2word``, and ``word2ind`` into the folder notebook is in.
Run the cells through with ``shift + enter``. Also refer to comments in the notebook.

## Run web server
1. Change into ``lyrics/projects``
2. Create folder ``projects/web/static/web/dp-model``
4. Copy ``modelYYYYMMDD_X.h5``, ``ind2word``, ``word2ind`` into the above  folder.
5. Activate environment: ``source activate py3``
6. Run ``python manage.py runserver``
7. After server up running, go to browser: ``http://localhost:8000/web/index.html``
8. Type sentence into the text area and submit.
