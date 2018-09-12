# daev
Tool for reachability analysis and safety verification/falsification of differential algebraic equations.

This tool is run with Python 2.7 on Ubuntu 16.04

# tool installation

    clone tool from github: git clone https://github.com/verivital/daev.git

    add path to .bashrc file: for example: export PYTHONPATH="${PYTHONPATH}:/home/trhoangdung/tools/daev"

    install dependencies: sudo apt-get install python-pip
                          sudo pip install numpy scipy matplotlib
    **note that numpy scipy matplotlib need to be installed to python2.7

# run examples
    go in example folder, there are different examples available
    run example (for example): python2 interconnected_rotating_masses.py