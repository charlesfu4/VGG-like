import IPython.nbformat.current as nbf
nb = nbf.read(open('VGG16_cifar10.py', 'r'), 'py')
nbf.write(nb, open('VGG16_cifar19.ipynb', 'w'), 'ipynb')
