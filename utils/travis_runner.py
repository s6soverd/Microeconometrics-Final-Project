#!/usr/bin/env python
#"""This script manages all tasks for the TRAVIS build server."""
#import subprocess as sp
#import glob

#if __name__ == '__main__':

    #for notebook in glob.glob('*.ipynb'):
        #cmd = ' jupyter nbconvert --execute {}  --ExecutePreprocessor.timeout=-1'.format(notebook)
        #sp.check_call(cmd, shell=True)
        
        
#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import subprocess as sp
import glob

if __name__ == '__main__':
    cmd = 'jupyter nbconvert --debug --ExecutePreprocessor.enabled=True --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1 --to html replication.ipynb'
    sp.check_call(cmd, shell=True)
    #for notebook in glob.glob('*.ipynb'):
        #cmd = ' jupyter nbconvert --execute {}  --ExecutePreprocessor.timeout=-1'.format(notebook)
        #cmd = ' jupyter nbconvert --debug {} --ExecutePreprocessor.enabled=True --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1'.format(notebook)
        #sp.check_call(cmd, shell=True)
